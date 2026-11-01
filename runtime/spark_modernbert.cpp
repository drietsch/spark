// Full ModernBERT mathematical program. No PyTorch/Transformers inference calls.
// Coefficients are runtime float32 data; BLAS implements dense numerical functions.
#include <algorithm>
#include <array>
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
#include <vector>
extern "C" void cblas_sgemm(int,int,int,int,int,int,float,const float*,int,const float*,int,float,float*,int);
namespace {
thread_local std::string error;
struct Span {float*p=nullptr;size_t size=0;};
struct Layer {Span an,ab,qkv,qb,aw,awb,mn,mb,wi,bi,wo,bo,rope;int active=0;};
struct Model {
 void*map=MAP_FAILED;size_t bytes=0;int h=0,i=0,l=0,heads=0,vocab=0,maxseq=0,window=0,every=0,tied=0,act=0,headact=0;float eps=0;
 Span embedding,en,eb,fn,fb,hd,hb,hn,hnb,decoder,db;std::vector<Layer>layers;
 ~Model(){if(map!=MAP_FAILED)munmap(map,bytes);}
};
void require(bool ok,const char*message){if(!ok)throw std::runtime_error(message);}
uint32_t u32(const unsigned char*p){uint32_t v;std::memcpy(&v,p,4);return v;}
float f32(const unsigned char*p){float v;std::memcpy(&v,p,4);return v;}
std::unique_ptr<Model> load(const char*path){
 require(path,"missing weight path");uint32_t endian=1;require(*reinterpret_cast<unsigned char*>(&endian)==1&&sizeof(float)==4,"requires little-endian float32 host");
 int fd=open(path,O_RDONLY);require(fd>=0,"cannot open converted weights");struct stat st{};
 if(fstat(fd,&st)!=0||st.st_size<68||st.st_size>2LL*1024*1024*1024){close(fd);throw std::runtime_error("invalid model allocation budget");}
 auto m=std::make_unique<Model>();m->bytes=st.st_size;m->map=mmap(nullptr,m->bytes,PROT_READ|PROT_WRITE,MAP_PRIVATE,fd,0);close(fd);
 require(m->map!=MAP_FAILED,"cannot map converted weights");auto*p=static_cast<unsigned char*>(m->map);
 require(std::memcmp(p,"SPRKMB01",8)==0&&u32(p+8)==1,"unsupported model binary");
 m->h=u32(p+12);m->i=u32(p+16);m->l=u32(p+20);m->heads=u32(p+24);m->vocab=u32(p+28);m->maxseq=u32(p+32);m->window=u32(p+36);m->every=u32(p+40);m->tied=u32(p+44);m->act=u32(p+48);m->headact=u32(p+52);m->eps=f32(p+56);
 require(m->h>=1&&m->h<=2048&&m->i>=1&&m->i<=8192&&m->l>=1&&m->l<=64&&m->heads>=1&&m->heads<=64&&m->h%m->heads==0&&(m->h/m->heads)%2==0&&m->vocab>=1&&m->vocab<=131072,"invalid model geometry");
 require(m->maxseq>=1&&m->maxseq<=8192&&m->window>=1&&m->window<=8192&&m->every>=1&&m->every<=64&&m->tied>=0&&m->tied<=1&&m->act>=0&&m->act<=1&&m->headact>=0&&m->headact<=1&&std::isfinite(m->eps)&&m->eps>0&&m->eps<1&&std::isfinite(f32(p+60))&&f32(p+60)>0&&std::isfinite(f32(p+64))&&f32(p+64)>0,"invalid execution constants");
 size_t cursor=68;auto span=[&](size_t n){require(n<=(m->bytes-cursor)/4,"truncated tensor payload");Span s{reinterpret_cast<float*>(p+cursor),n};cursor+=4*n;for(size_t k=0;k<n;++k)require(std::isfinite(s.p[k]),"nonfinite model coefficient");return s;};
 size_t h=m->h,i=m->i,v=m->vocab;m->embedding=span(v*h);m->en=span(h);m->eb=span(h);
 for(int n=0;n<m->l;++n){Layer a;a.an=span(h);a.ab=span(h);a.qkv=span(3*h*h);a.qb=span(3*h);a.aw=span(h*h);a.awb=span(h);a.mn=span(h);a.mb=span(h);a.wi=span(2*i*h);a.bi=span(2*i);a.wo=span(h*i);a.bo=span(h);a.rope=span(h/m->heads/2);a.active=m->i;m->layers.push_back(a);}
 m->fn=span(h);m->fb=span(h);m->hd=span(h*h);m->hb=span(h);m->hn=span(h);m->hnb=span(h);m->decoder=m->tied?m->embedding:span(v*h);m->db=span(v);
 require(cursor==m->bytes,"unexpected model trailing data");return m;
}
void linear(const std::vector<float>&x,int rows,int ins,int outs,Span w,Span b,std::vector<float>&y,int stride=0){
 y.resize(size_t(rows)*outs);cblas_sgemm(101,111,112,rows,outs,ins,1.0f,x.data(),ins,w.p,stride?stride:ins,0.0f,y.data(),outs);
 for(int r=0;r<rows;++r)for(int j=0;j<outs;++j)y[size_t(r)*outs+j]+=b.p[j];
}
void norm(const std::vector<float>&x,int rows,int h,Span w,Span b,float eps,std::vector<float>&y){
 y.resize(x.size());for(int r=0;r<rows;++r){const float*row=x.data()+size_t(r)*h;double sum=0;for(int k=0;k<h;++k)sum+=row[k];float mean=float(sum/h);double var=0;
 for(int k=0;k<h;++k){double d=double(row[k])-mean;var+=d*d;}float inv=1.0f/std::sqrt(float(var/h)+eps);
 for(int k=0;k<h;++k)y[size_t(r)*h+k]=((row[k]-mean)*inv)*w.p[k]+b.p[k];}
}
float gelu(float x,int mode){return mode==0?0.5f*x*(1.0f+std::erf(x*0.7071067811865475244f)):0.5f*x*(1.0f+std::tanh(0.7978845608028653559f*(x+0.044715f*x*x*x)));}
void residual(std::vector<float>&x,const std::vector<float>&delta){for(size_t j=0;j<x.size();++j){x[j]+=delta[j];require(std::isfinite(x[j]),"nonfinite mathematical state");}}
void attention(Model&m,Layer&a,int layer,const std::vector<float>&x,const int32_t*mask,const int32_t*pos,int t,std::vector<float>&out){
 int h=m.h,d=h/m.heads,half=d/2;std::vector<float>qkv;linear(x,t,h,3*h,a.qkv,a.qb,qkv);
 for(int r=0;r<t;++r)for(int head=0;head<m.heads;++head)for(int k=0;k<half;++k){
  float angle=float(pos[r])*a.rope.p[k];float c=std::cos(angle),s=std::sin(angle);
  for(int group=0;group<2;++group){size_t base=size_t(r)*3*h+group*h+head*d;float first=qkv[base+k],second=qkv[base+half+k];qkv[base+k]=first*c-second*s;qkv[base+half+k]=second*c+first*s;}}
 std::vector<float>ctx(size_t(t)*h),q(size_t(t)*d),k(q.size()),v(q.size()),weights(size_t(t)*t),headout(size_t(t)*d);
 bool local=layer%m.every!=0;float scale=1.0f/std::sqrt(float(d));
 for(int head=0;head<m.heads;++head){
  for(int r=0;r<t;++r)for(int j=0;j<d;++j){q[size_t(r)*d+j]=qkv[size_t(r)*3*h+head*d+j];k[size_t(r)*d+j]=qkv[size_t(r)*3*h+h+head*d+j];v[size_t(r)*d+j]=qkv[size_t(r)*3*h+2*h+head*d+j];}
  cblas_sgemm(101,111,112,t,t,d,1.0f,q.data(),d,k.data(),d,0.0f,weights.data(),t);
  for(int row=0;row<t;++row){float maximum=-std::numeric_limits<float>::infinity();
   for(int col=0;col<t;++col){size_t at=size_t(row)*t+col;weights[at]*=scale;if(!mask[col]||(local&&std::abs(row-col)>m.window/2))weights[at]=-std::numeric_limits<float>::infinity();maximum=std::max(maximum,weights[at]);}
   double denominator=0;for(int col=0;col<t;++col){size_t at=size_t(row)*t+col;float value=std::isfinite(maximum)?std::exp(weights[at]-maximum):0.f;weights[at]=value;denominator+=value;}
   for(int col=0;col<t;++col)weights[size_t(row)*t+col]=denominator>0?float(weights[size_t(row)*t+col]/denominator):0.f;
  }
  cblas_sgemm(101,111,111,t,d,t,1.0f,weights.data(),t,v.data(),d,0.0f,headout.data(),d);
  for(int row=0;row<t;++row)std::copy(headout.data()+size_t(row)*d,headout.data()+size_t(row+1)*d,ctx.data()+size_t(row)*h+head*d);
 }
 linear(ctx,t,h,h,a.aw,a.awb,out);
}
void run(Model&m,const int32_t*ids,const int32_t*mask,const int32_t*pos,int t,const int32_t*queries,int n,float*logits,float*hidden){
 require(ids&&mask&&pos&&queries&&logits&&t>=1&&t<=m.maxseq&&n>=1&&n<=t,"invalid input pointers/sequence budget");
 int present=0;for(int r=0;r<t;++r){require(ids[r]>=0&&ids[r]<m.vocab&&mask[r]>=0&&mask[r]<=1&&pos[r]>=0&&pos[r]<m.maxseq,"invalid token/mask/position");present+=mask[r];}require(present>0,"empty attention context");
 for(int q=0;q<n;++q)require(queries[q]>=0&&queries[q]<t&&mask[queries[q]],"invalid prediction position");
 std::vector<float>x(size_t(t)*m.h),temp,delta,projection,product;
 for(int r=0;r<t;++r)std::copy(m.embedding.p+size_t(ids[r])*m.h,m.embedding.p+size_t(ids[r]+1)*m.h,x.data()+size_t(r)*m.h);
 norm(x,t,m.h,m.en,m.eb,m.eps,temp);x.swap(temp);
 for(int layer=0;layer<m.l;++layer){auto&a=m.layers[layer];if(layer==0)temp=x;else norm(x,t,m.h,a.an,a.ab,m.eps,temp);
  attention(m,a,layer,temp,mask,pos,t,delta);residual(x,delta);norm(x,t,m.h,a.mn,a.mb,m.eps,temp);
  int active=a.active;product.resize(size_t(t)*active);
  if(active==m.i){linear(temp,t,m.h,2*m.i,a.wi,a.bi,projection);
   for(int r=0;r<t;++r)for(int j=0;j<active;++j)product[size_t(r)*active+j]=gelu(projection[size_t(r)*2*m.i+j],m.act)*projection[size_t(r)*2*m.i+m.i+j];}
  else {std::vector<float>gate;linear(temp,t,m.h,active,a.wi,a.bi,projection);
   Span gw{a.wi.p+size_t(m.i)*m.h,size_t(m.i)*m.h},gb{a.bi.p+m.i,size_t(m.i)};linear(temp,t,m.h,active,gw,gb,gate);
   for(size_t j=0;j<product.size();++j)product[j]=gelu(projection[j],m.act)*gate[j];}
  linear(product,t,active,m.h,a.wo,a.bo,delta,m.i);residual(x,delta);
 }
 norm(x,t,m.h,m.fn,m.fb,m.eps,temp);x.swap(temp);if(hidden)std::copy(x.begin(),x.end(),hidden);
 std::vector<float>selected(size_t(n)*m.h);for(int q=0;q<n;++q)std::copy(x.data()+size_t(queries[q])*m.h,x.data()+size_t(queries[q]+1)*m.h,selected.data()+size_t(q)*m.h);
 linear(selected,n,m.h,m.h,m.hd,m.hb,temp);for(float&v:temp)v=gelu(v,m.headact);norm(temp,n,m.h,m.hn,m.hnb,m.eps,selected);
 linear(selected,n,m.h,m.vocab,m.decoder,m.db,temp);for(float value:temp)require(std::isfinite(value),"nonfinite decoder output");std::copy(temp.begin(),temp.end(),logits);
}
}
extern "C" const char* spark_mb_error(){return error.c_str();}
extern "C" void* spark_mb_load(const char*path){try{error.clear();return load(path).release();}catch(const std::exception&e){error=e.what();return nullptr;}}
extern "C" void spark_mb_destroy(void*p){delete static_cast<Model*>(p);}
extern "C" int spark_mb_meta(void*p,int32_t*out){if(!p||!out)return 0;auto&m=*static_cast<Model*>(p);int values[]={m.h,m.i,m.l,m.heads,m.vocab,m.maxseq};std::copy(values,values+6,out);return 1;}
extern "C" int spark_mb_run(void*p,const int32_t*ids,const int32_t*mask,const int32_t*pos,int t,const int32_t*q,int n,float*out,float*hidden){try{require(p,"model closed");run(*static_cast<Model*>(p),ids,mask,pos,t,q,n,out,hidden);return 1;}catch(const std::exception&e){error=e.what();return 0;}}
extern "C" int spark_mb_set_width(void*p,int layer,int width){try{require(p,"model closed");auto&m=*static_cast<Model*>(p);require(layer>=0&&layer<m.l&&width>=1&&width<=m.i,"invalid active function width");m.layers[layer].active=width;return 1;}catch(const std::exception&e){error=e.what();return 0;}}
extern "C" int spark_mb_set_coefficient(void*p,int layer,int tensor,uint64_t index,float value){try{require(p,"model closed");auto&m=*static_cast<Model*>(p);require(layer>=0&&layer<m.l&&tensor>=0&&tensor<4&&std::isfinite(value)&&std::abs(value)<=1e4,"invalid live function coefficient");auto&a=m.layers[layer];Span spans[]={a.wi,a.bi,a.wo,a.bo};require(index<spans[tensor].size,"coefficient index out of range");spans[tensor].p[index]=value;return 1;}catch(const std::exception&e){error=e.what();return 0;}}
