"""SPARK M23: safe, explainable, finite Horn-clause logical question answering.

The model is a mathematical inference engine over facts and learned/user-supplied
logical rules, NOT a language model or a natural-language semantic oracle.
English support is intentionally a documented controlled grammar.

Inference uses forward chaining on positive antecedents, explicit negative heads,
and an open-world four-valued consequence relation (true/false/both/unknown).
Nothing is inferred from the absence of evidence.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

FORM = 'spark-logical-program-v1'
MAX_RULES=128
MAX_FACTS=4000
MAX_DERIVED=25000
MAX_ROUNDS=40
MAX_BODY=3
MAX_VARIABLES=4


def variable(name: str) -> bool:
    return bool(re.fullmatch(r'[A-Z]',name) or re.fullmatch(r'\?[a-z]',name))


def canon(s: str) -> str:
    return re.sub(r'\s+', ' ', s.strip().lower())


def singular(word: str) -> str:
    word=canon(word)
    irregular={'people':'person','children':'child','mice':'mouse','men':'man','women':'woman','geese':'goose','wolves':'wolf','leaves':'leaf','cities':'city','stories':'story'}
    if word in irregular:return irregular[word]
    if word.endswith('ies') and len(word)>4:return word[:-3]+'y'
    if word.endswith(('ches','shes','sses','xes','zes')):return word[:-2]
    if word.endswith('s') and not word.endswith(('ss','us','is')):return word[:-1]
    return word


@dataclass(frozen=True)
class Atom:
    pred: str
    args: tuple[str,...]
    neg: bool=False

    def __post_init__(self):
        if not re.fullmatch(r'[a-z][a-z0-9_]*',self.pred):raise ValueError('invalid predicate')
        if not 1<=len(self.args)<=2:raise ValueError('predicate arity must be one or two')
        if not all(re.fullmatch(r'(?:[a-z][a-z0-9_]*|[A-Z]|\?[a-z])',a) for a in self.args):
            raise ValueError('invalid entity or variable')

    def ground(self):return not any(variable(a) for a in self.args)
    def label(self):
        return ('not ' if self.neg else '')+self.pred+'('+', '.join(self.args)+')'


@dataclass(frozen=True)
class Rule:
    head: Atom
    body: tuple[Atom,...]
    source: str

    def __post_init__(self):
        if not 1<=len(self.body)<=MAX_BODY:raise ValueError('rule requires 1–3 antecedents')
        if any(b.neg for b in self.body):raise ValueError('negated antecedents unsupported; no negation-as-failure')
        names={a for atom in self.body for a in atom.args if variable(a)}
        if len(names)>MAX_VARIABLES:raise ValueError('too many variables')
        if not all(a in names for a in self.head.args if variable(a)):
            raise ValueError('unsafe rule: head variable absent from body')


@dataclass(frozen=True)
class Evidence:
    source: str
    premises: tuple[Atom,...]=()


@dataclass(frozen=True)
class Answer:
    status: str
    question: str
    supporting: tuple[str,...]
    opposing: tuple[str,...]

    def as_dict(self):
        return {'answer':self.status,'question':self.question,
                'proof':list(self.supporting),'counterproof':list(self.opposing)}


def _entity(s: str) -> str:
    s=s.strip().strip('.,;:')
    s=re.sub(r'^(?:the|a|an)\s+','',s,flags=re.I)
    if variable(s):return s
    s=canon(s).replace(' ','_')
    if not re.fullmatch(r'[a-z][a-z0-9_]*',s):raise ValueError(f'invalid name: {s!r}')
    return s


def _predicate(s: str) -> str:
    s=canon(s).replace(' ','_')
    if not re.fullmatch(r'[a-z][a-z0-9_]*',s):raise ValueError(f'invalid relation: {s!r}')
    return s


def parse_atom(text: str) -> Atom:
    """Recognize a restricted, compositional set of English propositions.

    Examples: X is a cat; Alice is not happy; X is a parent of Y;
    Alice is taller than Bob; Alice likes Bob; light is on.
    """
    text=text.strip().rstrip(' .?!').strip()
    if not text:raise ValueError('empty proposition')
    if re.search(r'\band\b', text, flags=re.I):
        raise ValueError('conjunction is only supported in if-then antecedents')
    # is (not) a NAME of OTHER
    m=re.fullmatch(r'(.+?)\s+is\s+(not\s+)?(?:a|an)\s+([\w-]+)\s+of\s+(.+)',text,flags=re.I)
    if m:
        p=''.join([singular(m.group(3)),'_of'])
        return Atom(_predicate(p),(_entity(m.group(1)),_entity(m.group(4))),bool(m.group(2)))
    # is (not) RELATION than OTHER
    m=re.fullmatch(r'(.+?)\s+is\s+(not\s+)?([\w-]+)\s+than\s+(.+)',text,flags=re.I)
    if m:return Atom(_predicate(m.group(3)+'_than'),(_entity(m.group(1)),_entity(m.group(4))),bool(m.group(2)))
    # is (not) (a/an) PROPERTY, including 'is on'
    m=re.fullmatch(r'(.+?)\s+is\s+(not\s+)?(?:(?:a|an)\s+)?([\w-]+)',text,flags=re.I)
    if m:return Atom(_predicate(singular(m.group(3))),(_entity(m.group(1)),),bool(m.group(2)))
    # Subject likes / admires / supports Object; generic English binary verbs in third-person.
    m=re.fullmatch(r'(.+?)\s+([a-z]+s)\s+(.+)',text,flags=re.I)
    if m and m.group(2).lower() not in ('is','was','has'):
        return Atom(_predicate(m.group(2)),(_entity(m.group(1)),_entity(m.group(3))))
    raise ValueError(f'unsupported controlled-English proposition: {text!r}')


def parse_question(question: str) -> Atom:
    s=question.strip().rstrip(' ?.').strip()
    m=re.fullmatch(r'Is\s+(.+?)\s+(not\s+)?(?:a|an)\s+([\w-]+)\s+of\s+(.+)',s,flags=re.I)
    if m:return parse_atom(f'{m.group(1)} is {m.group(2) or ""}a {m.group(3)} of {m.group(4)}')
    m=re.fullmatch(r'Is\s+(.+?)\s+(not\s+)?([\w-]+)\s+than\s+(.+)',s,flags=re.I)
    if m:return parse_atom(f'{m.group(1)} is {m.group(2) or ""}{m.group(3)} than {m.group(4)}')
    m=re.fullmatch(r'Is\s+(.+?)\s+(not\s+)?(?:(?:a|an)\s+)?([\w-]+)',s,flags=re.I)
    if m:return parse_atom(f'{m.group(1)} is {m.group(2) or ""}{m.group(3)}')
    m=re.fullmatch(r'Does\s+(.+?)\s+([a-z]+)\s+(.+)',s,flags=re.I)
    if m:
        verb=m.group(2).lower()
        if verb.endswith('y') and not verb.endswith(('ay','ey','oy','uy')):verb=verb[:-1]+'ies'
        elif verb.endswith(('ch','sh','ss','x','z')):verb+='es'
        else:verb+='s'
        return parse_atom(f'{m.group(1)} {verb} {m.group(3)}')
    raise ValueError(f'unsupported controlled-English question: {question!r}')


def parse_statement(text: str) -> Atom | Rule:
    text=text.strip().rstrip(' .').strip()
    m=re.fullmatch(r'(?:All|Every)\s+([a-z-]+)\s+(?:are|is)\s+(?:a\s+|an\s+)?([a-z-]+)',text,flags=re.I)
    if m:
        v='X';a=singular(m.group(1));b=singular(m.group(2))
        return Rule(Atom(_predicate(b),(v,)),(Atom(_predicate(a),(v,)),),text)
    m=re.fullmatch(r'No\s+([a-z-]+)\s+(?:are|is)\s+(?:a\s+|an\s+)?([a-z-]+)',text,flags=re.I)
    if m:
        v='X';a=singular(m.group(1));b=singular(m.group(2))
        return Rule(Atom(_predicate(b),(v,),True),(Atom(_predicate(a),(v,)),),text)
    m=re.fullmatch(r'If\s+(.+?)\s*,?\s+then\s+(.+)',text,flags=re.I)
    if m:
        conditions=tuple(parse_atom(s) for s in re.split(r'\s+and\s+',m.group(1),flags=re.I))
        return Rule(parse_atom(m.group(2)),conditions,text)
    return parse_atom(text)


def _unify(pattern:Atom,fact:Atom,env:dict[str,str]):
    if pattern.pred!=fact.pred or pattern.neg!=fact.neg or len(pattern.args)!=len(fact.args):return None
    result=env.copy()
    for name,value in zip(pattern.args,fact.args):
        if variable(name):
            if name in result and result[name]!=value:return None
            result[name]=value
        elif name!=value:return None
    return result


def _substitute(atom:Atom,env:dict[str,str])->Atom:
    return Atom(atom.pred,tuple(env[a] if variable(a) else a for a in atom.args),atom.neg)


class LogicProgram:
    def __init__(self,*,max_derived=MAX_DERIVED):
        if not 1<=max_derived<=MAX_DERIVED:raise ValueError('invalid deduction budget')
        self.facts:dict[Atom,Evidence]={}
        self.rules:list[Rule]=[]
        self.max_derived=max_derived
        self._derived:dict[Atom,Evidence]|None=None

    def add(self,sentence:str):
        stmt=parse_statement(sentence)
        if isinstance(stmt,Atom):
            if not stmt.ground():raise ValueError('asserted fact cannot contain variables')
            if len(self.facts)>=MAX_FACTS:raise ValueError('too many facts')
            self.facts[stmt]=Evidence(sentence.strip())
        else:
            if len(self.rules)>=MAX_RULES:raise ValueError('too many rules')
            self.rules.append(stmt)
        self._derived=None
        return stmt

    def add_many(self,text:str):
        for piece in re.split(r'[\n.;]+',text):
            if piece.strip():self.add(piece.strip())
        return self

    def solve(self):
        if self._derived is not None:return self._derived
        allfacts=dict(self.facts)
        for _ in range(MAX_ROUNDS):
            by_pred={}
            for a in allfacts:by_pred.setdefault((a.pred,a.neg),[]).append(a)
            additions={}
            for rule in self.rules:
                partials=[({},())]
                for body in rule.body:
                    new=[]
                    for env,premises in partials:
                        for fact in by_pred.get((body.pred,body.neg),()):
                            unified=_unify(body,fact,env)
                            if unified is not None:new.append((unified,premises+(fact,)))
                    partials=new
                    if not partials:break
                for env,premises in partials:
                    head=_substitute(rule.head,env)
                    if head not in allfacts and head not in additions:
                        additions[head]=Evidence(rule.source,premises)
            if not additions:
                self._derived=allfacts
                return allfacts
            if len(allfacts)+len(additions)>self.max_derived:
                raise RuntimeError('reasoning fact budget exceeded; answer withheld')
            allfacts.update(additions)
        raise RuntimeError('reasoning iteration budget exceeded; answer withheld')

    def _proof(self,atom:Atom,seen=None):
        seen=set() if seen is None else seen
        if atom in seen:return []
        seen.add(atom)
        info=self.solve()[atom]
        out=[]
        for prem in info.premises:out.extend(self._proof(prem,seen))
        out.append((f'{atom.label()} — given: {info.source}' if not info.premises else
                    f'{atom.label()} — by rule: {info.source}'))
        return out

    def ask(self,question:str)->Answer:
        query=parse_question(question)
        if not query.ground():raise ValueError('questions must refer to concrete entities')
        pos=query
        neg=Atom(query.pred,query.args,not query.neg)
        evidence=self.solve()
        supported=pos in evidence
        opposed=neg in evidence
        status='inconsistent' if supported and opposed else 'yes' if supported else 'no' if opposed else 'unknown'
        return Answer(status,question,tuple(self._proof(pos)) if supported else (),
                      tuple(self._proof(neg)) if opposed else ())

    def save(self,path:Path):
        """Store learned rule ASTs, not only their English source strings."""
        def atom_json(a):return {'pred':a.pred,'args':list(a.args),'neg':a.neg}
        path=Path(path)
        path.parent.mkdir(parents=True,exist_ok=True)
        rules=[{'head':atom_json(r.head),'body':[atom_json(b) for b in r.body],
                'source':r.source} for r in self.rules]
        path.write_text(json.dumps({'format':FORM,'facts':[v.source for v in self.facts.values()],
            'rules':rules},sort_keys=True,indent=2)+'\n')

    @classmethod
    def load(cls,path:Path):
        obj=json.loads(Path(path).read_text())
        if not isinstance(obj,dict) or set(obj)!={'format','facts','rules'} or obj['format']!=FORM:
            raise ValueError('invalid saved reasoning program')
        if (type(obj['facts']) is not list or type(obj['rules']) is not list or
                len(obj['facts'])>MAX_FACTS or len(obj['rules'])>MAX_RULES):
            raise ValueError('invalid saved program lengths')
        p=cls()
        for s in obj['facts']:
            if type(s) is not str or len(s)>2048 or not isinstance(parse_statement(s),Atom):
                raise ValueError('saved fact invalid')
            p.add(s)
        def read_atom(value):
            if not isinstance(value,dict) or set(value)!={'pred','args','neg'} or \
                    type(value['pred']) is not str or type(value['args']) is not list or \
                    type(value['neg']) is not bool or not all(type(x) is str for x in value['args']):
                raise ValueError('invalid serialized atom')
            return Atom(value['pred'],tuple(value['args']),value['neg'])
        for r in obj['rules']:
            if type(r) is str:  # M23 early-format compatibility
                stmt=parse_statement(r)
                if not isinstance(stmt,Rule):raise ValueError('saved rule invalid')
            elif isinstance(r,dict) and set(r)=={'head','body','source'} and \
                    type(r['source']) is str and len(r['source'])<=2048 and type(r['body']) is list:
                stmt=Rule(read_atom(r['head']),tuple(read_atom(a) for a in r['body']),r['source'])
            else:raise ValueError('invalid serialized rule')
            p.rules.append(stmt)
        return p

    def learn_transitivity(self, relation: str, labels: list[tuple[str,bool]]):
        """Learn an optional reusable Horn rule from labelled logical queries.

        Hypotheses are {existing rules, existing rules + transitivity}. A
        negative label means 'not entailed' during selection, not closed-world
        falsity at inference. Scores use TRAIN labels only; callers must reserve
        independent examples and unseen entities for testing. This is bounded
        structural rule induction, not natural-language rule discovery.
        """
        template=parse_atom('X '+relation+' Y')
        if template.neg or len(template.args)!=2 or template.args!=('X','Y'):
            raise ValueError('transitivity candidate requires binary positive relation')
        if len(labels)<3 or len(labels)>300 or any(type(y) is not bool for _,y in labels):
            raise ValueError('need 3–300 labelled true/false examples')
        for q,_ in labels:
            atom=parse_question(q)
            if atom.pred!=template.pred or len(atom.args)!=2 or atom.neg:
                raise ValueError('labelled example outside relation')
        clause=Rule(Atom(template.pred,('X','Z')),
                    (Atom(template.pred,('X','Y')),Atom(template.pred,('Y','Z'))),
                    f'learned transitivity of {template.pred} from labelled queries')
        def errors(p):
            return sum((p.ask(q).status in ('yes','inconsistent'))!=label for q,label in labels)
        baseline=errors(self)
        trial=LogicProgram(max_derived=self.max_derived)
        trial.facts=self.facts.copy()
        trial.rules=self.rules+[clause]
        after=errors(trial)
        selected=after<baseline
        if selected:
            if len(self.rules)>=MAX_RULES:raise ValueError('too many learned rules')
            self.rules.append(clause)
            self._derived=None
        return {'selected':selected,'baseline_errors':baseline,
                'candidate_errors':after,'examples':len(labels),'relation':template.pred}
