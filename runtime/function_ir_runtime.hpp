// Generic numerical kernels for the SPARK function language. No model topology.
#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fcntl.h>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <unordered_set>
#include <vector>
extern "C" void cblas_dgemm(int,int,int,int,int,int,double,const double*,int,const double*,int,double,double*,int);
namespace spark_function {
inline thread_local std::string last_error;
inline void ensure(bool value,const char* message){if(!value)throw std::runtime_error(message);}
inline int64_t elements(const std::vector<int64_t>& shape){
 int64_t size=1;ensure(!shape.empty()&&shape.size()<=4,"invalid tensor rank");
 for(auto d:shape){ensure(d>0&&size<=(128LL<<20)/d,"tensor allocation budget exceeded");size*=d;}return size;
}
struct Tensor {
 std::shared_ptr<std::vector<double>> storage;
 std::vector<int64_t> shape,stride;int64_t offset=0;
 Tensor()=default;
 explicit Tensor(std::vector<int64_t> dims):storage(std::make_shared<std::vector<double>>(elements(dims))),shape(std::move(dims)){
  stride.resize(shape.size());int64_t step=1;for(int i=int(shape.size())-1;i>=0;--i){stride[i]=step;step*=shape[i];}}
 int64_t size()const{return elements(shape);}
 double* data()const{return storage->data()+offset;}
 int64_t at(int64_t flat)const{int64_t out=offset;for(int i=int(shape.size())-1;i>=0;--i){out+=(flat%shape[i])*stride[i];flat/=shape[i];}return out;}
 double get(int64_t flat)const{return (*storage)[at(flat)];}
 bool contiguous()const{int64_t step=1;for(int i=int(shape.size())-1;i>=0;--i){if(shape[i]!=1&&stride[i]!=step)return false;step*=shape[i];}return true;}
};
inline Tensor packed(const Tensor& x){if(x.contiguous())return x;Tensor out(x.shape);for(int64_t i=0;i<x.size();++i)out.data()[i]=x.get(i);return out;}
inline Tensor input_tensor(const double* values,std::vector<int64_t> shape,int type){
 ensure(values,"missing input buffer");Tensor out(std::move(shape));for(int64_t i=0;i<out.size();++i){double v=values[i];
  ensure(std::isfinite(v),"nonfinite input");if(type==1)ensure(v==std::floor(v)&&std::abs(v)<=2147483647.,"invalid discrete index");
  if(type==2)ensure(v==0.||v==1.,"invalid Boolean value");out.data()[i]=v;}return out;
}
inline void check_bounds(const Tensor& x,int64_t upper){for(int64_t i=0;i<x.size();++i)ensure(x.get(i)>=0&&x.get(i)<upper,"index outside declared bounds");}
inline void check_unique(const Tensor& x){std::unordered_set<double> found;for(int64_t i=0;i<x.size();++i)ensure(found.insert(x.get(i)).second,"duplicate query index");}
inline void check_any(const Tensor& x){for(int64_t i=0;i<x.size();++i)if(x.get(i)!=0.)return;throw std::runtime_error("empty input selection");}
inline void check_selected(const Tensor& data,const Tensor& indices){check_bounds(indices,data.size());for(int64_t i=0;i<indices.size();++i)ensure(data.get(int64_t(indices.get(i)))!=0.,"query selects excluded input");}
inline void copy_output(const Tensor& x,double* out){for(int64_t i=0;i<x.size();++i){double v=x.get(i);ensure(std::isfinite(v),"nonfinite program result");out[i]=v;}}
struct Parameter {uint64_t offset,count;std::vector<int64_t> shape;bool mutable_value;int precision;};
struct Context {
 void* mapping=MAP_FAILED;size_t bytes=0;std::vector<Parameter> parameters;std::vector<Tensor> cache;
 std::vector<int> widths,capacities;
 Context(const char* path,std::vector<Parameter> specs,std::vector<int> w,std::vector<int> cap):parameters(std::move(specs)),cache(parameters.size()),widths(std::move(w)),capacities(std::move(cap)){
  uint32_t endian=1;ensure(*reinterpret_cast<unsigned char*>(&endian)==1&&sizeof(float)==4&&sizeof(double)==8,"little-endian IEEE coefficient storage required");
  int fd=open(path,O_RDONLY);ensure(fd>=0,"cannot open function coefficients");struct stat info{};
  if(fstat(fd,&info)!=0){close(fd);throw std::runtime_error("cannot stat coefficients");}
  uint64_t expected=0;for(auto& p:parameters){ensure(p.offset==expected&&p.count==uint64_t(elements(p.shape))&&(p.precision==4||p.precision==8),"invalid coefficient binding");expected+=p.count*p.precision;}
  if(info.st_size<0||uint64_t(info.st_size)!=expected||expected>(2ULL<<30)){close(fd);throw std::runtime_error("coefficient payload size differs from compiled program");}
  bytes=expected;if(bytes)mapping=mmap(nullptr,bytes,PROT_READ|PROT_WRITE,MAP_PRIVATE,fd,0);close(fd);
  ensure(bytes==0||mapping!=MAP_FAILED,"coefficient mapping failed");
  for(auto& p:parameters)for(uint64_t i=0;i<p.count;++i)if(!std::isfinite(read(p,i))){munmap(mapping,bytes);mapping=MAP_FAILED;throw std::runtime_error("nonfinite coefficient");}
 }
 ~Context(){if(mapping!=MAP_FAILED)munmap(mapping,bytes);}
 double read(const Parameter& p,uint64_t index)const{auto* raw=static_cast<const unsigned char*>(mapping)+p.offset+index*p.precision;
  if(p.precision==4){float value;std::memcpy(&value,raw,4);return value;}double value;std::memcpy(&value,raw,8);return value;}
 Tensor parameter(size_t index){ensure(index<parameters.size(),"unknown coefficient binding");if(!cache[index].storage){auto& p=parameters[index];Tensor tensor(p.shape);
  for(uint64_t i=0;i<p.count;++i)tensor.data()[i]=read(p,i);cache[index]=tensor;}return cache[index];}
 void width(int function,int value){ensure(function>=0&&size_t(function)<widths.size()&&value>=1&&value<=capacities[function],"invalid live function width");widths[function]=value;}
 void coefficient(int index,uint64_t position,double value){ensure(index>=0&&size_t(index)<parameters.size(),"unknown live coefficient");auto& p=parameters[index];
  ensure(p.mutable_value&&position<p.count&&std::isfinite(value)&&std::abs(value)<=1e4,"invalid live coefficient update");
  auto* raw=static_cast<unsigned char*>(mapping)+p.offset+position*p.precision;
  if(p.precision==4){float rounded=float(value);std::memcpy(raw,&rounded,4);}else std::memcpy(raw,&value,8);
  if(cache[index].storage)cache[index].data()[position]=read(p,position);}
};
template<class F> Tensor mapped(const std::vector<Tensor>& inputs,F function){
 size_t rank=0;for(auto& x:inputs)rank=std::max(rank,x.shape.size());std::vector<int64_t> shape(rank,1);
 for(auto& x:inputs)for(size_t i=0;i<x.shape.size();++i){size_t axis=rank-x.shape.size()+i;ensure(shape[axis]==1||x.shape[i]==1||shape[axis]==x.shape[i],"incompatible broadcast");shape[axis]=std::max(shape[axis],x.shape[i]);}
 Tensor out(shape);std::vector<int64_t> index(rank);std::vector<double> args(inputs.size());
 for(int64_t flat=0;flat<out.size();++flat){int64_t rest=flat;for(int axis=int(rank)-1;axis>=0;--axis){index[axis]=rest%shape[axis];rest/=shape[axis];}
  for(size_t i=0;i<inputs.size();++i){const auto& x=inputs[i];int64_t at=x.offset;for(size_t axis=0;axis<x.shape.size();++axis)if(x.shape[axis]!=1)at+=index[rank-x.shape.size()+axis]*x.stride[axis];args[i]=(*x.storage)[at];}
  out.data()[flat]=function(args.data(),index.data());}return out;
}
inline Tensor reduced(const Tensor& x,int axis,bool maximum){
 auto shape=x.shape;int64_t count=shape[axis];shape[axis]=1;Tensor out(shape);
 for(int64_t i=0;i<out.size();++i){int64_t rest=i,base=x.offset;for(int a=int(shape.size())-1;a>=0;--a){base+=(rest%shape[a])*x.stride[a];rest/=shape[a];}
  double value=maximum?-std::numeric_limits<double>::infinity():0.;for(int64_t j=0;j<count;++j){double v=(*x.storage)[base+j*x.stride[axis]];value=maximum?std::max(value,v):value+v;}out.data()[i]=value;}return out;
}
template<class F> Tensor mapped_reduced(const std::vector<Tensor>& inputs,int axis,bool maximum,F function){
 size_t rank=0;for(const auto& x:inputs)rank=std::max(rank,x.shape.size());std::vector<int64_t> shape(rank,1);
 for(const auto& x:inputs)for(size_t i=0;i<x.shape.size();++i){size_t a=rank-x.shape.size()+i;ensure(shape[a]==1||x.shape[i]==1||shape[a]==x.shape[i],"incompatible fused broadcast");shape[a]=std::max(shape[a],x.shape[i]);}
 int64_t count=shape[axis];shape[axis]=1;Tensor out(shape);std::vector<int64_t> index(rank),base(inputs.size());std::vector<double> args(inputs.size());
 std::vector<std::vector<int64_t>> strides(inputs.size(),std::vector<int64_t>(rank));
 for(size_t i=0;i<inputs.size();++i)for(size_t a=0;a<inputs[i].shape.size();++a)if(inputs[i].shape[a]!=1)strides[i][rank-inputs[i].shape.size()+a]=inputs[i].stride[a];
 for(int64_t flat=0;flat<out.size();++flat){int64_t rest=flat;for(int a=int(rank)-1;a>=0;--a){index[a]=rest%shape[a];rest/=shape[a];}
  for(size_t i=0;i<inputs.size();++i){base[i]=inputs[i].offset;for(size_t a=0;a<rank;++a)base[i]+=index[a]*strides[i][a];}
  double total=maximum?-std::numeric_limits<double>::infinity():0.;
  for(int64_t j=0;j<count;++j){index[axis]=j;for(size_t i=0;i<inputs.size();++i)args[i]=(*inputs[i].storage)[base[i]+j*strides[i][axis]];
   double value=function(args.data(),index.data());total=maximum?std::max(total,value):total+value;}
  out.data()[flat]=total;}return out;
}
inline Tensor transposed(Tensor x,const std::vector<int64_t>& axes){auto shape=x.shape,stride=x.stride;for(size_t i=0;i<axes.size();++i){x.shape[i]=shape[axes[i]];x.stride[i]=stride[axes[i]];}return x;}
inline Tensor reshaped(const Tensor& value,const std::vector<int64_t>& shape){ensure(elements(shape)==value.size(),"reshape changes element count");Tensor x=packed(value);x.shape=shape;x.stride.resize(shape.size());int64_t step=1;for(int i=int(shape.size())-1;i>=0;--i){x.stride[i]=step;step*=shape[i];}return x;}
inline Tensor sliced(Tensor x,int axis,int64_t start,int64_t size){ensure(start>=0&&size>=1&&start+size<=x.shape[axis],"slice outside tensor");x.offset+=start*x.stride[axis];x.shape[axis]=size;return x;}
inline Tensor selected(Tensor x,int axis,int64_t index){ensure(index>=0&&index<x.shape[axis],"map index outside tensor");x.offset+=index*x.stride[axis];x.shape.erase(x.shape.begin()+axis);x.stride.erase(x.stride.begin()+axis);return x;}
inline Tensor concatenated(const Tensor& a,const Tensor& b,int axis){auto shape=a.shape;shape[axis]+=b.shape[axis];Tensor out(shape);
 for(int64_t flat=0;flat<out.size();++flat){int64_t rest=flat,ia=a.offset,ib=b.offset;bool first=true;for(int i=int(shape.size())-1;i>=0;--i){int64_t at=rest%shape[i];rest/=shape[i];if(i==axis){first=at<a.shape[i];ia+=at*a.stride[i];ib+=(at-a.shape[i])*b.stride[i];}else{ia+=at*a.stride[i];ib+=at*b.stride[i];}}
  out.data()[flat]=first?(*a.storage)[ia]:(*b.storage)[ib];}return out;
}
inline Tensor gathered(const Tensor& data,const Tensor& indices){ensure(data.shape.size()==2&&indices.shape.size()==1,"invalid row gather");check_bounds(indices,data.shape[0]);Tensor out({indices.size(),data.shape[1]});
 for(int64_t row=0;row<indices.size();++row){int64_t selected_row=int64_t(indices.get(row));for(int64_t col=0;col<data.shape[1];++col)out.data()[row*data.shape[1]+col]=(*data.storage)[data.offset+selected_row*data.stride[0]+col*data.stride[1]];}return out;
}
inline Tensor contracted(const Tensor& left,const Tensor& right){
 ensure(left.shape.size()==2&&right.shape.size()==2&&left.shape[1]==right.shape[0],"invalid contraction");
 Tensor a=left,b=right;bool at=a.stride[0]==1&&a.stride[1]>=a.shape[0],bt=b.stride[0]==1&&b.stride[1]>=b.shape[0];
 if(!at&&a.stride[1]!=1)a=packed(a);if(!bt&&b.stride[1]!=1)b=packed(b);
 int64_t m=a.shape[0],n=b.shape[1],k=a.shape[1];Tensor out({m,n});
 cblas_dgemm(101,at?112:111,bt?112:111,int(m),int(n),int(k),1.,a.data(),int(at?a.stride[1]:a.stride[0]),b.data(),int(bt?b.stride[1]:b.stride[0]),0.,out.data(),int(n));return out;
}
template<class F> Tensor mapped_call(const std::vector<Tensor>& inputs,const std::vector<int64_t>& axes,F function){
 int64_t count=0;for(size_t i=0;i<axes.size();++i)if(axes[i]>=0){if(!count)count=inputs[i].shape[axes[i]];ensure(count==inputs[i].shape[axes[i]],"map extent mismatch");}ensure(count>0,"empty mapped call");Tensor out;
 for(int64_t i=0;i<count;++i){std::vector<Tensor> args;for(size_t j=0;j<inputs.size();++j)args.push_back(axes[j]<0?inputs[j]:selected(inputs[j],int(axes[j]),i));Tensor value=function(args);
  if(!i){auto shape=value.shape;shape.insert(shape.begin(),count);out=Tensor(shape);}ensure(out.size()==count*value.size(),"mapped result shape mismatch");
  for(int64_t j=0;j<value.size();++j)out.data()[i*value.size()+j]=value.get(j);}return out;
}
} // namespace spark_function
