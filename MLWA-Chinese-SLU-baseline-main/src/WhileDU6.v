Require Import Coq.Strings.String.
Require Import Coq.ZArith.ZArith.
Require Import Coq.micromega.Psatz.


Require Import SetsClass.SetsClass. Import SetsNotation.
Require Import PL.SyntaxInCoq.
Require Import compcert.lib.Integers.
Require Import PL.PracticalDenotations.
Local Open Scope bool.
Local Open Scope string.
Local Open Scope Z.
Local Open Scope sets.

Require Import Coq.Lists.List.
Import ListNotations.

Definition var_name: Type := string.
Import Lang_WhileD.
Import Lang_While.
Module Lang_WhileDU6.


Inductive type : Type :=
| TInt : type
| TPointer (t: type) : type
| TStruct (x: string) : type
| TUnion (x: string) : type
with men_var : Type :=
| MVar  (men_type : type) (var_name:  string) :  men_var.

Inductive expr_type : Type :=
| EConst (n: Z) (ty : type) : expr_type
| EVar (x: var_name) (ty:type): expr_type
| EBinop (op: binop) (e1 e2: expr_type)(ty:type) : expr_type
| EUnop (op: unop) (e: expr_type)(ty:type) : expr_type
| EDeref (e: expr_type)(ty:type) : expr_type
| EAddrOf (e: expr_type) (ty:type): expr_type
| EStructMember (x:expr_type) (field: var_name) (ty:type): expr_type  (* Access struct member *)
| EUnionMember (x:expr_type) (field: var_name)(ty:type) : expr_type  (* Access union member *)
| EPoniter_Struct_Member (x:expr_type) (field: var_name)(ty:type) : expr_type  (* Access poniter member *)
| EPoniter_Union_Member (x:expr_type) (field: var_name)(ty:type) : expr_type  (* Access poniter member *).

(*把类型环境拆出去*)
Record state: Type := {
  env: var_name -> int64;
  mem: int64 -> option val;
  type_env : var_name -> type;
}.

Notation "s '.(env)'" := (env s) (at level 1).
Notation "s '.(mem)'" := (mem s) (at level 1).
Notation "s '.(type_env)'" := (type_env s) (at level 1).


Record type_env_state: Type := {
    struct_info : string -> list men_var;
    union_info :  string->list men_var;
    type_size : type -> Z;
    type_size_properties :
        type_size TInt = 1 ->
        (forall t : type, type_size (TPointer t) = 1) ->
        (forall x : string, type_size (TStruct x) = 
            fold_left (fun acc field =>
                          match field with
                          | MVar ty name => acc + type_size ty
                          end) (struct_info x) 0) ->
        (forall x : string, type_size (TUnion x) =
            fold_left (fun acc field =>
                          match field with
                          | MVar ty name => Z.max acc (type_size ty)
                          end) (union_info x) 0) ->
                          True
}.


Module EDenote_type.

Record EDenote_type: Type := {
  type_nrm: type_env_state->state -> int64 -> Prop;
  type_err: type_env_state->state -> Prop;
}.

End EDenote_type.

Import EDenote_type.

Notation "x '.(type_nrm)'" := (EDenote_type.type_nrm x)
  (at level 1, only printing).

Notation "x '.(type_err)'" := (EDenote_type.type_err x)
  (at level 1, only printing).

Ltac any_nrm x := exact (EDenote_type.type_nrm x).

Ltac any_err x := exact (EDenote_type.type_err x).

Notation "x '.(type_nrm)'" := (ltac:(any_nrm x))
  (at level 1, only parsing).

Notation "x '.(type_err)'" := (ltac:(any_err x))
  (at level 1, only parsing).


Definition NotStructOrUnion (ty1 ty2:type): Prop := 
              forall su , ty1 <> TStruct su /\  
                          ty1 <> TUnion su /\ 
                          ty2 <> TStruct su /\ 
                          ty2 <> TUnion su.
  
Definition type_equal (ty1 ty2: type): Prop := 
    ty1 = ty2.


Definition arith_sem1_nrm
    (Zfun: Z -> Z -> Z)
    (D1 D2: state -> int64 -> Prop)
    (s: state)
    (i: int64): Prop :=
  exists i1 i2,
  D1 s i1 /\ D2 s i2 /\
  arith_compute1_nrm Zfun i1 i2 i.

Definition arith_sem1_err
  (Zfun: Z -> Z -> Z)
  (D1 D2: state -> int64 -> Prop)
  (s: state): Prop :=
exists i1 i2,
D1 s i1 /\ D2 s i2 /\
arith_compute1_err Zfun i1 i2.

Definition arith_sem1 Zfun (D1 D2: EDenote_type): EDenote_type :=
{|
type_nrm := fun t  =>  (arith_sem1_nrm Zfun (D1.(type_nrm) t) (D2.(type_nrm) t));
type_err := fun t  => arith_sem1_err Zfun (D1.(type_nrm) t) (D2.(type_nrm) t) ;
|}.

Definition arith_sem2_nrm
             (int64fun: int64 -> int64 -> int64)
             (D1 D2: state -> int64 -> Prop)
             (s: state)
             (i: int64): Prop :=
  exists i1 i2,
    D1 s i1 /\ D2 s i2 /\
    arith_compute2_nrm int64fun i1 i2 i.

Definition arith_sem2_err
             (D1 D2: state -> int64 -> Prop)
             (s: state): Prop :=
  exists i1 i2,
    D1 s i1 /\ D2 s i2 /\
    arith_compute2_err i1 i2.

Definition arith_sem2 int64fun (D1 D2: EDenote_type): EDenote_type :=
  {|
    type_nrm := fun t =>arith_sem2_nrm int64fun (D1.(type_nrm) t) (D2.(type_nrm) t);
    type_err := fun t => D1.(type_err) t ∪ D2.(type_err) t ∪
                          arith_sem2_err (D1.(type_nrm) t) (D2.(type_nrm) t);
  |}.

Definition cmp_sem_nrm
(c: comparison)
(D1 D2: state -> int64 -> Prop)
(s: state)
(i: int64): Prop :=
exists i1 i2,
D1 s i1 /\ D2 s i2 /\ cmp_compute_nrm c i1 i2 i.

Definition cmp_sem c (D1 D2: EDenote_type): EDenote_type :=
{|
type_nrm := fun t  =>   (cmp_sem_nrm c (D1.(type_nrm) t) (D2.(type_nrm) t));
type_err := fun t  =>   (D1.(type_err) t) ∪ (D2.(type_err) t);
|}.


Definition neg_sem_nrm
  (D1: state -> int64 -> Prop)
  (s: state)
  (i: int64): Prop :=
exists i1, D1 s i1 /\ neg_compute_nrm i1 i.

Definition neg_sem_err
             (D1: state -> int64 -> Prop)
             (s: state): Prop :=
  exists i1, D1 s i1 /\ neg_compute_err i1.
  

Definition neg_sem (D1: EDenote_type): EDenote_type :=
  {|
    type_err := fun t => ((D1.(type_err) t) ∪ neg_sem_err (D1.(type_nrm) t) ) ;
    type_nrm := fun t =>  neg_sem_nrm (D1.(type_nrm) t);
  |}.


Definition not_sem_nrm
  (D1: state -> int64 -> Prop)
  (s: state)
  (i: int64): Prop :=
exists i1, D1 s i1 /\ not_compute_nrm i1 i.

Definition not_sem (D1: EDenote_type): EDenote_type :=
{|
type_nrm := fun t => not_sem_nrm (D1.(type_nrm) t);
type_err := fun t => D1.(type_err) t;
|}.

Definition and_sem_nrm
    (D1 D2: state -> int64 -> Prop)
    (s: state)
    (i: int64): Prop :=
    exists i1,
    D1 s i1 /\
    (SC_and_compute_nrm i1 i \/
    NonSC_and i1 /\
    exists i2,
    D2 s i2 /\ NonSC_compute_nrm i2 i).

    Definition and_sem_err
    (D1: state -> int64 -> Prop)
    (D2: state -> Prop)
    (s: state): Prop :=
    exists i1,
    D1 s i1 /\ NonSC_and i1 /\ D2 s.

Definition and_sem (D1 D2: EDenote_type): EDenote_type :=
{|
type_nrm := fun t =>  and_sem_nrm ( D1.(type_nrm) t) (D2.(type_nrm) t);
type_err := fun t => (D1.(type_err) t )∪ and_sem_err (D1.(type_nrm) t) (D2.(type_err) t);
|}.


Definition or_sem_nrm
(D1 D2: state -> int64 -> Prop)
(s: state)
(i: int64): Prop :=
exists i1,
D1 s i1 /\
(SC_or_compute_nrm i1 i \/
NonSC_or i1 /\
exists i2,
D2 s i2 /\ NonSC_compute_nrm i2 i).

Definition or_sem_err
(D1: state -> int64 -> Prop)
(D2: state -> Prop)
(s: state): Prop :=
exists i1,
D1 s i1 /\ NonSC_or i1 /\ D2 s.

Definition or_sem (D1 D2: EDenote_type) : EDenote_type :=
{|
type_nrm := fun t => or_sem_nrm (D1.(type_nrm) t)( D2.(type_nrm) t );
type_err := fun t => D1.(type_err) t ∪ or_sem_err (D1.(type_nrm) t) ( D2.(type_err) t);
|}.

Definition unop_sem (op: unop) (D: EDenote_type): EDenote_type :=
match op with
| ONeg =>  D
| ONot => not_sem D
end.

Definition binop_sem (op: binop) (D1 D2: EDenote_type) : EDenote_type :=
match op with
| OOr => or_sem D1 D2
| OAnd => and_sem D1 D2
| OLt => cmp_sem Clt D1 D2
| OLe => cmp_sem Cle D1 D2
| OGt => cmp_sem Cgt D1 D2
| OGe => cmp_sem Cge D1 D2
| OEq => cmp_sem Ceq D1 D2
| ONe => cmp_sem Cne D1 D2
| OPlus => arith_sem1 Z.add D1 D2 
| OMinus => arith_sem1 Z.sub D1 D2 
| OMul => arith_sem1 Z.mul D1 D2 
| ODiv => arith_sem2 Int64.divs D1 D2
| OMod => arith_sem2 Int64.mods D1 D2
end.

Definition const_sem (n: Z) : EDenote_type :=
    {|
      type_nrm := fun t s i =>
               i = Int64.repr n /\
               Int64.min_signed <= n <= Int64.max_signed ;
      type_err := fun t s =>
               n < Int64.min_signed \/
               n > Int64.max_signed ;
    |}.

(** 『解引用』表达式既可以用作右值也可以用作左值。其作为右值是的语义就是原先我们
    定义的『解引用』语义。*)

Definition deref_sem_nrm
    (D1: state -> int64 -> Prop)
    (s: state)
    (i: int64): Prop :=
    exists i1, D1 s i1 /\ s.(mem) i1 = Some (Vint i).

Definition deref_sem_err
    (D1: state -> int64 -> Prop)
    (s: state): Prop :=
    exists i1,
    D1 s i1 /\
    (s.(mem) i1 = None \/ s.(mem) i1 = Some Vuninit).

Definition deref_sem_r (D1: EDenote_type): EDenote_type :=
        {|
        type_nrm := fun t => deref_sem_nrm (D1.(type_nrm) t);
        type_err := fun t => D1.(type_err) t ∪ (deref_sem_err (D1.(type_nrm) t));
        |}.
   

Definition var_sem_l (X: var_name): EDenote_type :=
        {|
        type_nrm := fun t s i => s.(env) X = i;
        type_err := ∅;
        |}.
  
    Definition var_sem_r (X: var_name) (ty : type) : EDenote_type :=
          {|
            type_nrm :=   (fun  t s i => s.(type_env) X = ty ) ∩ (deref_sem_r (var_sem_l X)).(type_nrm);
            type_err :=   (fun  t s => s.(type_env) X <> ty) ∪(deref_sem_r (var_sem_l X)).(type_err) ;
            
          
          
            |}.
        

Definition False_sem: EDenote_type :=
    {|
    type_nrm := ∅;
    type_err := Sets.full;
    |}.

(*辅助函数：查找结构体中某个字段的类型*)
Fixpoint calculate_type  (field:string) (fields:list men_var) : option type := 
  match fields with
  | [] => None
  | MVar ty f :: rest =>
    if string_dec f field
      then Some ty
    else calculate_type  field rest
  end.

  (* 辅助函数：计算结构体中字段的偏移量 *)
Fixpoint calculate_offset (t:type_env_state) (field: string) (fields: list men_var) (offset: Z) : option Z :=
    match fields with
    | [] => None (* 未找到字段，返回 None *)
    | MVar ty f :: rest =>
      if string_dec f field
      then Some offset (* 找到字段，返回当前偏移量 *)
      else calculate_offset t field rest (offset + t.(type_size) ty ) (* 继续查找下一个字段 *)
    end.


    (* 辅助函数：查找结构体中字段的偏移量 *)
Definition find_field_offset (t : type_env_state) (field: string) (struct_fields: list men_var) : option Z :=
    calculate_offset t field struct_fields (0).

Definition noSU_var_sem_l (X: var_name): EDenote_type :=
{|
type_nrm := fun t s i => s.(env) X = i /\ (forall su_type, (s.(type_env) X) <> TStruct su_type /\ (s.(type_env) X) <> TUnion su_type);
type_err := ∅;
|}.
Definition EStructMember_sem_l (D1: EDenote_type) (struct_type: string) (field: string) (ty:type) : EDenote_type :=
{|
    type_nrm := (fun t s i => exists i' , (D1.(type_nrm) t s i' )/\ (
                                match find_field_offset t field (t.(struct_info) struct_type) with
                                (*查一下这个字段field在D1对应的struct Type里的偏移量*)
                                | Some offset => i = Int64.add i' (Int64.repr offset)
                                | None => False
                                end) /\ calculate_type  field (t.(struct_info) struct_type) = Some ty
                                    );
    type_err := fun t => D1.(type_err) t ∪ (fun s => exists i', (D1.(type_nrm) t s i') /\ 
                                (
                                match find_field_offset t  field (t.(struct_info) struct_type) with
                                | Some offset => False
                                | None => True
                                end
                                \/ calculate_type field (t.(struct_info) struct_type) <> Some ty
                                )
                    );
|}.


Definition EUnionMember_sem_l (D1: EDenote_type) (union_type: string) (field: string) (ty:type): EDenote_type :=
    {|
    type_nrm := (fun t s i => exists i' , (D1.(type_nrm) t s i' )/\ (
                                    match find_field_offset t field (t.(union_info) union_type) with
                                    (*查一下这个字段field在D1对应的union Type里的"偏移"量*)
                                    | Some offset => i = i' 
                                    | None => False
                                    end) /\ calculate_type  field (t.(union_info) union_type) = Some ty
                                    );
    type_err := fun t => D1.(type_err) t ∪ (fun s => exists i', (D1.(type_nrm) t s i') /\ (
                                    match find_field_offset t field ( (t.(union_info) union_type)) with
                                    | Some offset => False
                                    | None => True
                                    end 
                                    \/ calculate_type field (t.(union_info) union_type) <> Some ty
                                    ));
    |}. 


Definition struct_member_sem_r (x:EDenote_type) (struct_type: string)(field:string)(ty:type): EDenote_type :=
    {|
        type_nrm :=  fun t s i => exists i', (x.(type_nrm) t s i') /\ (
                                                    match find_field_offset t field (t.(struct_info) struct_type) with
                                                    (*查一下这个字段field在x对应的struct Type里的偏移量*)
                                                    | None => False
                                                    | Some offset => s.(mem) (Int64.add i' (Int64.repr offset)) = Some (Vint i)
                                                    end  /\ calculate_type  field (t.(struct_info) struct_type) = Some ty);
        type_err := fun t=>x.(type_err) t ∪ (fun s => exists i', (x.(type_nrm) t s i') /\ (
                                                    match find_field_offset t field (t.(struct_info) struct_type) with
                                                    | None => True
                                                    | Some offset => s.(mem) (Int64.add i' (Int64.repr offset)) = None \/ s.(mem) (Int64.add i' (Int64.repr offset)) = Some (Vuninit)
                                                    end 
                                                    \/ calculate_type field (t.(struct_info) struct_type) <> Some ty
                                                    ));
    |}.
    
Definition union_member_sem_r (x:EDenote_type) (union_type: string)(field:string)(ty:type): EDenote_type :=
    {|
        type_nrm :=  fun t s i => exists i', (x.(type_nrm) t s i') /\ (
                                                    match find_field_offset t field (t.(union_info) union_type) with
                                                    (*查一下这个字段field在x对应的union Type里的"偏移"量*)
                                                    None => False
                                                    | Some offset => s.(mem) i' = Some (Vint i)
                                                    end  /\ calculate_type  field (t.(union_info) union_type) = Some ty);
        type_err := fun t=> x.(type_err) t  ∪ (fun s => exists i', (x.(type_nrm) t s i') /\ (
                                                    match find_field_offset t field (t.(union_info) union_type) with
                                                    | None => True
                                                    | Some offset => s.(mem) i' = None \/ s.(mem) i' = Some (Vuninit)
                                                    end
                                                    \/ calculate_type field (t.(union_info) union_type) <> Some ty
                                                    ));
    |}.

Definition expr_type_extract (e:expr_type): type :=
        match e with
        | EConst n ty => ty
        | EVar x ty => ty
        | EBinop op e1 e2 ty => ty
        | EUnop op e ty => ty
        | EDeref e ty => ty
        | EAddrOf e ty => ty
        | EStructMember x field ty => ty
        | EUnionMember x field ty => ty
        | EPoniter_Struct_Member x field ty => ty
        | EPoniter_Union_Member x field ty => ty
        end.

    Fixpoint eval_r (e: expr_type): EDenote_type :=
        match e with

        | EConst n ty =>
            match ty with
            | TInt => const_sem n
            | _ => False_sem
            end

        | EVar X ty =>
            match ty with
            | TInt => var_sem_r X ty
            | TPointer _ => var_sem_r X ty
            | _ => False_sem
            end
            (*基于区分左右值的思路，右值只能是TInt 或者 TPointer*)
      
        | EBinop op e1 e2 ty=>
          (*此处我们认为只有TInt 和 TPointer 可以进行二元运算*)
            match expr_type_extract e1 as t1 with
              | TStruct _ => False_sem
              | TUnion _ => False_sem
              | TInt => 
                match expr_type_extract e2 with
                | TStruct _ => False_sem
                | TUnion _ => False_sem
                | TInt =>
                  match ty with
                  | TInt => binop_sem op (eval_r e1) (eval_r e2)
                  | _ => False_sem
                  end
                | TPointer _ =>
                  match ty with
                  | TInt => binop_sem op (eval_r e1) (eval_r e2)
                  | _ => False_sem
                  end
                end
              | TPointer pointer_type => 
                match expr_type_extract e2 with
                | TStruct _ => False_sem
                | TUnion _ => False_sem
                | TInt =>
                  match ty with
                  | TInt => binop_sem op (eval_r e1) (eval_r e2)
                  | _ => False_sem
                  end
                | TPointer pointer_type =>
                  match ty with
                  | TPointer pointer_type => binop_sem op (eval_r e1) (eval_r e2)
                    (*这里值得注意的是，遇到了俩pointer，那么必须一致指向类型*)
                  | _ => False_sem
                end
              end
            end 

        | EUnop op e1 ty=>
            match ty  with
            | TStruct _ => False_sem
            | TUnion _ => False_sem
            (*老样子，右值不允许SU*)
            | TInt =>
                match expr_type_extract e1 with
                | TInt=> unop_sem op (eval_r e1)
                | _ => False_sem
                end
            | TPointer Pointer_type => 
                match expr_type_extract e1 with
                | TPointer Pointer_type=> unop_sem op (eval_r e1)
                | _ => False_sem
            end
            end
      
        | EDeref e1 ty=>
            match ty  with
            | TStruct _ => False_sem
            | TUnion _ => False_sem
            (*老样子，右值不允许SU*)
            | TInt =>
                match expr_type_extract e1 with
                | TInt => deref_sem_r  (eval_r e1)
                | _ => False_sem
                end
            | TPointer Pointer_type => 
                match expr_type_extract e1 with
                | TPointer ty => deref_sem_r  (eval_r e1)
                | _ => False_sem
            end
            end
        
          
            (*eval_r e1 返回了什么？返回了的是一个Edenote ， 是 state->int64， 是返回了int64，那么你随便解析吧*)
      
        | EAddrOf e1 ty=>
          match ty with
          | TPointer pointed_type =>
            match expr_type_extract e1 with
            | pointed_type => eval_l e1
            end
          | _ => False_sem
            end
            (*这里的e1必须是个左值，那么我们就需要一个eval_l*)
            
            
            (*这里后面会判断，是个Var，那没关系，我们总是能通过state.env 找到这个变量的位置
              是个Dref？也没关系，反向回去好了。
            *)
        | EStructMember x field ty => 
          match ty with
          | TStruct _ => False_sem
          | TUnion _ => False_sem 
          | _ => 
            match expr_type_extract x with
            | TStruct struct_type =>   struct_member_sem_r (eval_l x) struct_type field ty
            | _ => False_sem
            end
          end

        | EUnionMember x field ty=>
            match ty with
            | TStruct _ => False_sem
            | TUnion _ => False_sem
            | _ => 
              match expr_type_extract x with
              | TUnion union_type => union_member_sem_r (eval_l x) union_type field ty
              | _ => False_sem
              end
            end  
      
        |EPoniter_Struct_Member x field ty=>
          match ty with
          | TStruct _ => False_sem
          | TUnion _ => False_sem
          | _ => 
            match expr_type_extract x with
            | TPointer pointed_type =>
              match pointed_type with
              | TStruct struct_type => struct_member_sem_r (eval_r x) struct_type field ty
              | _ => False_sem
              end
            | _ => False_sem
            end
          end
        
        | EPoniter_Union_Member x field ty=>
          match ty with
          | TStruct _ => False_sem
          | TUnion _ => False_sem
          | _ => 
            match expr_type_extract x with
            | TPointer pointed_type =>
              match pointed_type with
              | TUnion union_type => union_member_sem_r (eval_r x) union_type field ty
              | _ => False_sem
              end
            | _ => False_sem
            end
          end
        end
        with eval_l (e: expr_type): EDenote_type :=
        (*好，让我们想想能对什么东西计算左值
          1. EVar x ty （无条件）
          2. EDeref x ty （无条件）
          3. EStructMember x field ty
              这个玩意儿。。。。。的话，需要x的左值+ field在x这个struct中的偏移量
          4. EUnionMember x field ty
          5. EPoniter_Struct_Member x field ty
          6. EPoniter_Union_Member x field ty
        *)
        match e with
        | EVar X ty=>
            var_sem_l X
        | EDeref e1 ty =>
            eval_r e1
        | EStructMember x field ty =>
            match expr_type_extract x with 
            | TStruct struct_type => EStructMember_sem_l (eval_l x) struct_type field ty
            | _ => False_sem
            end
        | EUnionMember x field ty=>
            match expr_type_extract x with
            | TUnion union_type => EUnionMember_sem_l (eval_l x) union_type field ty
            | _ => False_sem
            end
            
        | EPoniter_Struct_Member x field ty=>
          (*对于 x->y 来说，这玩意儿的左值
            就是说x是个指针，指向的是个struct
            那么对应的地址应该是 x的右值 + field在x这个struct中的偏移量
          *)
            match expr_type_extract x with
            | TPointer pointed_type =>
              match pointed_type with
              | TStruct struct_type => EStructMember_sem_l (eval_r x) struct_type field ty
              | _ => False_sem
              end
            | _ => False_sem
            end
        | EPoniter_Union_Member x field ty=>
            match expr_type_extract x with
            | TPointer pointed_type =>
              match pointed_type with
              | TUnion union_type => EUnionMember_sem_l (eval_r x) union_type field ty
              | _ => False_sem
              end
            | _ => False_sem
            end
        | _ =>
            {| type_nrm := ∅; type_err := Sets.full; |}
        end.


Definition test_true (D: EDenote_type):
        type_env_state -> state -> state -> Prop :=
        fun t =>
          Rels.test
            (fun s  =>
              exists i , D.(type_nrm) t s i /\ Int64.signed i <> 0).

Definition test_false  (D: EDenote_type):
    type_env_state->state -> state -> Prop := fun t=>
          Rels.test (fun s =>  D.(type_nrm) t s (Int64.repr 0)).

Module CDenote_type.

Record CDenote_type: Type := {
          type_nrm:  type_env_state->state-> state -> Prop ;
          type_err:  type_env_state->state -> Prop;
          type_inf:  type_env_state->state -> Prop
        }.        
End CDenote_type.

Import CDenote_type.
Definition SU_state_staysame (s1 s2: type_env_state): Prop :=

  s1.(struct_info) = s2.(struct_info) /\
  s1.(union_info) = s2.(union_info) /\
  s1.(type_size) = s2.(type_size).


Notation "x '.(type_nrm)'" := (CDenote_type.type_nrm x)
  (at level 1, only printing).

Notation "x '.(type_err)'" := (CDenote_type.type_err x)
  (at level 1, only printing).

Ltac any_nrm x ::=
  match type of x with
  | EDenote_type => exact (EDenote_type.type_nrm x)
  | CDenote_type => exact (CDenote_type.type_nrm x)
  end.

Ltac any_err x ::=
  match type of x with
  | EDenote_type => exact (EDenote_type.type_err x)
  | CDenote_type => exact (CDenote_type.type_err x)
  end.

Definition skip_sem: CDenote_type :=
  {|
    type_nrm := fun t=>Rels.id;
    type_err := ∅;
    type_inf := ∅;
  |}.

Definition seq_sem (D1 D2: CDenote_type): CDenote_type :=
  {|
    type_nrm :=fun t=> (D1.(type_nrm) t )∘ (D2.(type_nrm) t);
    type_err := fun t=>D1.(type_err) t ∪ (D1.(type_nrm) t ∘ (D2.(type_err) t));
    type_inf :=fun t=> D1.(type_inf) t ∪ ((D1.(type_nrm) t)∘( D2.(type_inf) t));
  |}.

Definition if_sem
             (D0: EDenote_type)
             (D1 D2: CDenote_type): CDenote_type :=
  {|
    type_nrm:= fun t=>
            ( (test_true  D0 t)  ∘ (D1.(type_nrm) t)) ∪
           ( (test_false  D0 t) ∘ (D2.(type_nrm) t));
    type_err := fun t=> 
            (D0.(type_err) t) ∪
           ((test_true  D0 t) ∘ (D1.(type_err) t)) ∪
           ((test_false  D0 t )∘ (D2.(type_err) t));
    type_inf := fun t=> 
            ((test_true  D0 t) ∘ (D1.(type_inf) t)) ∪
           ((test_false  D0 t) ∘ (D2.(type_inf) t))
  |}.

Fixpoint boundedLB_nrm
          (t: type_env_state)
           (D0: EDenote_type)
           (D1: CDenote_type)
           (n: nat):
   state -> state -> Prop :=
  match n with
  | O => ∅
  | S n0 =>
      ((test_true D0 t ) ∘ (D1.(type_nrm) t) ∘ boundedLB_nrm t D0 D1 n0) ∪
      (test_false D0 t)
  end.

Fixpoint boundedLB_err
            (t: type_env_state)
           (D0: EDenote_type)
           (D1: CDenote_type)
           (n: nat): state -> Prop :=
  match n with
  | O => ∅
  | S n0 =>
     (test_true D0 t∘
        ((D1.(type_nrm) t ∘ boundedLB_err t D0 D1 n0) ∪
         D1.(type_err) t)) ∪
      D0.(type_err) t
  end.

Definition is_inf
              (t: type_env_state)
             (D0: EDenote_type)
             (D1: CDenote_type)
             (X: state -> Prop): Prop :=
  X ⊆ test_true D0 t ∘ (((D1.(type_nrm) t) ∘ X) ∪ D1.(type_inf) t).

Definition while_sem
             (D0: EDenote_type)
             (D1: CDenote_type): CDenote_type :=
  {|
    type_nrm := fun t=>⋃ (boundedLB_nrm t D0 D1);
    type_err := fun t =>⋃ (boundedLB_err t D0 D1);
    type_inf := fun t =>Sets.general_union (is_inf t D0 D1);
  |}.

(** 向地址赋值的语义与原先定义基本相同，只是现在需要规定所有变量的地址不被改变，
    而非所有变量的值不被改变。*)

Definition asgn_deref_sem_nrm
             (D1 D2: state -> int64 -> Prop)
             (s1 s2: state): Prop :=
  exists i1 i2,
    D1 s1 i1 /\
    D2 s1 i2 /\
    s1.(mem) i1 <> None /\
    s2.(mem) i1 = Some (Vint i2) /\
    (forall X, s1.(env) X = s2.(env) X /\ s1.(type_env) X = s2.(type_env) X) /\
    (forall p, i1 <> p -> s1.(mem) p = s2.(mem) p).

Definition asgn_deref_sem_err
             (D1: state -> int64 -> Prop)
             (s1: state): Prop :=
  exists i1,
    D1 s1 i1 /\
    s1.(mem) i1 = None.

Definition asgn_deref_sem
             (D1 D2: EDenote_type): CDenote_type :=
  {|
    type_nrm := fun t=>(asgn_deref_sem_nrm (D1.(type_nrm) t) (D2.(type_nrm) t));
    type_err := fun t=> D1.(type_err)  t ∪ D2.(type_err) t ∪
           asgn_deref_sem_err (D2.(type_nrm) t);
    type_inf := ∅;
  |}.

(** 变量赋值的行为可以基于此定义。*)





Definition asgn_var_sem
             (X: var_name)
             (D1: EDenote_type): CDenote_type :=
  
  asgn_deref_sem (noSU_var_sem_l X) D1.

Check 12.
(* 定义 int64 序列生成函数 *)
Definition Zseq (start len : Z) : list Z :=
  List.map Z.of_nat (List.seq (Z.to_nat start) (Z.to_nat len)).

Definition declare_sem
             (X: var_name)
             (ty: type): CDenote_type :=
  {|
    type_nrm := fun t s1 s2 =>
             (*首先原来不能有这个变量名*)
             forall i,
               s1.(env) X <> i /\
              (*使用一块空地址*)
              exists i', Int64.unsigned i'>=0 ->s2.(env) X = i' /\
                  
                (  forall k,     ( Int64.unsigned i')<=k<= (Int64.unsigned i' + (t.(type_size) ty)) -> 
                                           ((s1.(mem) ( Int64.repr k) = Some Vuninit)  /\
                                           (s2.(mem) ( Int64.repr k) = Some (Vint (Int64.repr 0))))
                                           )  /\
                (forall k': Z, k' < (Int64.unsigned i') \/ k' >= (Int64.unsigned i')+((t.(type_size) ty)) -> 
                                                  s1.(mem) (Int64.repr k') = s2.(mem) (Int64.repr k')) /\
            
                (forall X', X' <> X -> s1.(env) X' = s2.(env) X' /\ s1.(type_env) X' = s2.(type_env) X')
            ;
                  
                    

              (*这个地址不能被占用*)
              (*exists i', 
                (forall pos  ipos var, (In pos (Zseq (Int64.unsigned (s1.(env) var)) (t.(type_size ) (s1.(type_env) var))))
                                        /\ (In (Int64.unsigned ipos) (Zseq (Int64.unsigned i') (t.(type_size ) (ty))))
                                        /\ s1.(env) var = ipos 
                                                -> ipos<> Int64.repr pos)
              /\
                s2.(env) X = i' /\
                (s1.(mem) i' <> None) /\
                (forall X', X <> X' -> s1.(env) X' = s2.(env) X') /\
                (forall p, i' <> p -> s1.(mem) p = s2.(mem) p) /\
                (s2.(type_env) X = ty) 
                ;*)


                type_err:= fun t s =>exists i, s.(env) X = i;
                type_inf := ∅;
  |}.


(** 在递归定义的程序语句语义中，只会直接使用表达式用作右值时的值。*)
Inductive com: Type :=
| CSkip : com
| CAsgnVar (X: var_name) (e: expr_type)
| CAsgnDeref (e1 e2: expr_type)
| CSeq (c1 c2: com)
| CIf (e: expr_type) (c1 c2: com)
| CWhile (e: expr_type) (c: com)
| CDeclare(X: var_name) (ty: type)
| CAsgnBasedOnExp (X: expr_type) (e: expr_type) .


Definition False_sem_com: CDenote_type :=
  {|
    type_nrm := ∅;
    type_err := Sets.full;
    type_inf := ∅;
  |}.

(** 递归定义的程序语句语义。*)
Definition Pointer_Check (e1 :expr_type) (ty:type): Prop :=
  match expr_type_extract e1 with
  | TPointer ty => True
  | _ => False
  end.
  
Fixpoint eval_com (c: com): CDenote_type :=
  match c with
  | CSkip =>
      skip_sem
  | CAsgnVar X e =>
      asgn_var_sem X (eval_r e)
  | CAsgnDeref e1 e2 =>
      match Pointer_Check e1  (expr_type_extract e2) with
      | True => asgn_deref_sem (eval_r e1) (eval_r e2)
      end
  | CSeq c1 c2 =>
      seq_sem (eval_com c1) (eval_com c2)
  | CIf e c1 c2 =>
      if_sem (eval_r e) (eval_com c1) (eval_com c2)
  | CWhile e c1 =>
      while_sem (eval_r e) (eval_com c1)
  | CDeclare X ty =>
      declare_sem X ty
  | CAsgnBasedOnExp X e =>
      asgn_deref_sem (eval_l X) (eval_r e)
  end.


End Lang_WhileDU6.


Module WhileDmalloc.
Inductive expr_malloc : Type :=
| EConst (n: Z)  : expr_malloc
| EVar (x: var_name) : expr_malloc
| EBinop (op: binop) (e1 e2: expr_malloc) : expr_malloc
| EUnop (op: unop) (e: expr_malloc) : expr_malloc
| EDeref (e: expr_malloc) : expr_malloc
| EAddrOf (e: expr_malloc) : expr_malloc.

Record state: Type := {
  env: var_name -> int64;
  mem: int64 -> option val;
}.

Notation "s '.(env)'" := (env s) (at level 1).
Notation "s '.(mem)'" := (mem s) (at level 1).

Module EDenote.

Record EDenote: Type := {
  nrm: state -> int64 -> Prop;
  err: state -> Prop;
}.

End EDenote.

Import EDenote.

Notation "x '.(nrm)'" := (EDenote.nrm x)
  (at level 1, only printing).

Notation "x '.(err)'" := (EDenote.err x)
  (at level 1, only printing).

Ltac any_nrm x := exact (EDenote.nrm x).

Ltac any_err x := exact (EDenote.err x).

Notation "x '.(nrm)'" := (ltac:(any_nrm x))
  (at level 1, only parsing).

Notation "x '.(err)'" := (ltac:(any_err x))
  (at level 1, only parsing).

  Definition arith_sem1_nrm
  (Zfun: Z -> Z -> Z)
  (D1 D2: state -> int64 -> Prop)
  (s: state)
  (i: int64): Prop :=
exists i1 i2,
D1 s i1 /\ D2 s i2 /\
arith_compute1_nrm Zfun i1 i2 i.

Definition arith_sem1_err
  (Zfun: Z -> Z -> Z)
  (D1 D2: state -> int64 -> Prop)
  (s: state): Prop :=
exists i1 i2,
D1 s i1 /\ D2 s i2 /\
arith_compute1_err Zfun i1 i2.

Definition arith_sem1 Zfun (D1 D2: EDenote): EDenote :=
{|
nrm := arith_sem1_nrm Zfun D1.(nrm) D2.(nrm)    ;
err := D1.(err) ∪ D2.(err) ∪
arith_sem1_err Zfun D1.(nrm) D2.(nrm);
|}.



Definition arith_sem2_nrm
             (int64fun: int64 -> int64 -> int64)
             (D1 D2: state -> int64 -> Prop)
             (s: state)
             (i: int64): Prop :=
  exists i1 i2,
    D1 s i1 /\ D2 s i2 /\
    arith_compute2_nrm int64fun i1 i2 i.

Definition arith_sem2_err
             (D1 D2: state -> int64 -> Prop)
             (s: state): Prop :=
  exists i1 i2,
    D1 s i1 /\ D2 s i2 /\
    arith_compute2_err i1 i2.

Definition arith_sem2 int64fun (D1 D2: EDenote): EDenote :=
  {|
    nrm := arith_sem2_nrm int64fun D1.(nrm) D2.(nrm);
    err := D1.(err) ∪ D2.(err) ∪
           arith_sem2_err D1.(nrm) D2.(nrm);
  |}.

  Definition cmp_sem_nrm
  (c: comparison)
  (D1 D2: state -> int64 -> Prop)
  (s: state)
  (i: int64): Prop :=
exists i1 i2,
D1 s i1 /\ D2 s i2 /\ cmp_compute_nrm c i1 i2 i.

Definition cmp_sem c (D1 D2: EDenote): EDenote :=
{|
nrm := cmp_sem_nrm c D1.(nrm) D2.(nrm);
err := D1.(err) ∪ D2.(err);
|}.

Definition neg_sem_nrm
  (D1: state -> int64 -> Prop)
  (s: state)
  (i: int64): Prop :=
exists i1, D1 s i1 /\ neg_compute_nrm i1 i.



Definition neg_sem_err
             (D1: state -> int64 -> Prop)
             (s: state): Prop :=
  exists i1, D1 s i1 /\ neg_compute_err i1.

Definition neg_sem (D1: EDenote): EDenote :=
  {|
    nrm := neg_sem_nrm D1.(nrm);
    err := D1.(err) ∪ neg_sem_err D1.(nrm);
  |}.

Definition not_sem_nrm
             (D1: state -> int64 -> Prop)
             (s: state)
             (i: int64): Prop :=
  exists i1, D1 s i1 /\ not_compute_nrm i1 i.

Definition not_sem (D1: EDenote): EDenote :=
  {|
    nrm := not_sem_nrm D1.(nrm);
    err := D1.(err);
  |}.

  Definition and_sem_nrm
  (D1 D2: state -> int64 -> Prop)
  (s: state)
  (i: int64): Prop :=
exists i1,
D1 s i1 /\
(SC_and_compute_nrm i1 i \/
NonSC_and i1 /\
exists i2,
D2 s i2 /\ NonSC_compute_nrm i2 i).
Definition and_sem_err
             (D1: state -> int64 -> Prop)
             (D2: state -> Prop)
             (s: state): Prop :=
  exists i1,
    D1 s i1 /\ NonSC_and i1 /\ D2 s.

Definition and_sem (D1 D2: EDenote): EDenote :=
  {|
    nrm := and_sem_nrm D1.(nrm) D2.(nrm);
    err := D1.(err) ∪ and_sem_err D1.(nrm) D2.(err);
  |}.

  
Definition or_sem_nrm
(D1 D2: state -> int64 -> Prop)
(s: state)
(i: int64): Prop :=
exists i1,
D1 s i1 /\
(SC_or_compute_nrm i1 i \/
NonSC_or i1 /\
exists i2,
D2 s i2 /\ NonSC_compute_nrm i2 i).

Definition or_sem_err
(D1: state -> int64 -> Prop)
(D2: state -> Prop)
(s: state): Prop :=
exists i1,
D1 s i1 /\ NonSC_or i1 /\ D2 s.

Definition or_sem (D1 D2: EDenote) : EDenote :=
{|
nrm := or_sem_nrm D1.(nrm) D2.(nrm) ;
err := D1.(err) ∪ or_sem_err D1.(nrm) D2.(err);
|}.

Definition unop_sem (op: unop) (D: EDenote): EDenote :=
  match op with
  | ONeg =>  D
  | ONot => not_sem D
  end.

Definition binop_sem (op: binop) (D1 D2: EDenote) : EDenote :=
  match op with
  | OOr => or_sem D1 D2
  | OAnd => and_sem D1 D2
  | OLt => cmp_sem Clt D1 D2
  | OLe => cmp_sem Cle D1 D2
  | OGt => cmp_sem Cgt D1 D2
  | OGe => cmp_sem Cge D1 D2
  | OEq => cmp_sem Ceq D1 D2
  | ONe => cmp_sem Cne D1 D2
  | OPlus => arith_sem1 Z.add D1 D2 
  | OMinus => arith_sem1 Z.sub D1 D2 
  | OMul => arith_sem1 Z.mul D1 D2 
  | ODiv => arith_sem2 Int64.divs D1 D2
  | OMod => arith_sem2 Int64.mods D1 D2
  end.

(*这里认为常数指针和值在处理上是一样的，在使用时会因为type不同产生区别*)
  Definition const_sem (n: Z) : EDenote :=
    {|
      nrm := fun s i =>
               i = Int64.repr n /\
               Int64.min_signed <= n <= Int64.max_signed ;
      err := fun s =>
               n < Int64.min_signed \/
               n > Int64.max_signed ;
    |}.
  
(** 『解引用』表达式既可以用作右值也可以用作左值。其作为右值是的语义就是原先我们
    定义的『解引用』语义。*)

Definition deref_sem_nrm
(D1: state -> int64 -> Prop)
(s: state)
(i: int64): Prop :=
exists i1, D1 s i1 /\ s.(mem)  i1 = Some (Vint i).

(*这里的i1是地址*)

Definition deref_sem_err
(D1: state -> int64 -> Prop)
(s: state): Prop :=
exists i1,
D1 s i1 /\
(s.(mem) i1 = None \/ s.(mem) i1 = Some Vuninit).

Definition deref_sem_r (D1: EDenote): EDenote :=
{|
nrm := deref_sem_nrm D1.(nrm);
err := D1.(err) ∪ deref_sem_err D1.(nrm);
|}.



(** 当程序表达式为单个变量时，它也可以同时用作左值或右值。下面先定义其作为左值时
的存储地址。*)

Definition var_sem_l (X: var_name): EDenote :=
{|
nrm := fun s i => s.(env) X = i;
err := ∅;
|}.

(*
Definition noSU_var_sem_l (X: var_name): EDenote :=
{|
nrm := fun s i => s.(env) X = i /\ (s.(type_env) X) <> TStruct X /\ (s.(type_env) X) <> TUnion X;
err := ∅;
|}.*)

(** 基于此，可以又定义它作为右值时的值。*)
(** 基于此，可以又定义它作为右值时的值。*)



Definition var_sem_r (X: var_name): EDenote :=
  deref_sem_r (var_sem_l X).


  (**此处，定义一个False的语句**)

  Definition False_sem: EDenote :=
    {|
    nrm := ∅;
    err := Sets.full;
    |}.
    
    





  Fixpoint eval_r (e: expr_malloc): EDenote :=
    match e with
    | EConst n =>
        const_sem n
    | EVar X =>
        var_sem_r X
    | EBinop op e1 e2 =>
        binop_sem op (eval_r e1) (eval_r e2)
    | EUnop op e1 =>
        unop_sem op (eval_r e1)
    | EDeref e1 =>
        deref_sem_r (eval_r e1)
    | EAddrOf e1 =>
        eval_l e1
    end
  with eval_l (e: expr_malloc): EDenote :=
    match e with
    | EVar X =>
        var_sem_l X
    | EDeref e1 =>
        eval_r e1
    | _ =>
        {| nrm := ∅; err := Sets.full; |}
    end.


  Definition test_true (D: EDenote):
  state -> state -> Prop :=
  Rels.test
    (fun s =>
       exists i, D.(nrm) s i /\ Int64.signed i <> 0).

Definition test_false (D: EDenote):
  state -> state -> Prop :=
  Rels.test (fun s => D.(nrm) s (Int64.repr 0)).

Module CDenote.


Record CDenote: Type := {
  nrm: state -> state -> Prop;
  err: state -> Prop;
  inf: state -> Prop
}.

End CDenote.
Import CDenote.
(*
Definition SU_state_staysame (s1 s2: state): Prop :=

  s1.(struct_info) = s2.(struct_info) /\
  s1.(union_info) = s2.(union_info) /\
  s1.(type_size) = s2.(type_size).
*)




Notation "x '.(nrm)'" := (CDenote.nrm x)
  (at level 1, only printing).

Notation "x '.(err)'" := (CDenote.err x)
  (at level 1, only printing).

Ltac any_nrm x ::=
  match type of x with
  | EDenote => exact (EDenote.nrm x)
  | CDenote => exact (CDenote.nrm x)
  end.

Ltac any_err x ::=
  match type of x with
  | EDenote => exact (EDenote.err x)
  | CDenote => exact (CDenote.err x)
  end.

Definition skip_sem: CDenote :=
  {|
    nrm := Rels.id;
    err := ∅;
    inf := ∅;
  |}.

Definition seq_sem (D1 D2: CDenote): CDenote :=
  {|
    nrm := D1.(nrm) ∘ D2.(nrm);
    err := D1.(err) ∪ (D1.(nrm) ∘ D2.(err));
    inf := D1.(inf) ∪ (D1.(nrm) ∘ D2.(inf));
  |}.

Definition if_sem
             (D0: EDenote)
             (D1 D2: CDenote): CDenote :=
  {|
    nrm := (test_true D0 ∘ D1.(nrm)) ∪
           (test_false D0 ∘ D2.(nrm));
    err := D0.(err) ∪
           (test_true D0 ∘ D1.(err)) ∪
           (test_false D0 ∘ D2.(err));
    inf := (test_true D0 ∘ D1.(inf)) ∪
           (test_false D0 ∘ D2.(inf))
  |}.

Fixpoint boundedLB_nrm
           (D0: EDenote)
           (D1: CDenote)
           (n: nat):
  state -> state -> Prop :=
  match n with
  | O => ∅
  | S n0 =>
      (test_true D0 ∘ D1.(nrm) ∘ boundedLB_nrm D0 D1 n0) ∪
      (test_false D0)
  end.

Fixpoint boundedLB_err
           (D0: EDenote)
           (D1: CDenote)
           (n: nat): state -> Prop :=
  match n with
  | O => ∅
  | S n0 =>
     (test_true D0 ∘
        ((D1.(nrm) ∘ boundedLB_err D0 D1 n0) ∪
         D1.(err))) ∪
      D0.(err)
  end.

Definition is_inf
             (D0: EDenote)
             (D1: CDenote)
             (X: state -> Prop): Prop :=
  X ⊆ test_true D0 ∘ ((D1.(nrm) ∘ X) ∪ D1.(inf)).

Definition while_sem
             (D0: EDenote)
             (D1: CDenote): CDenote :=
  {|
    nrm := ⋃ (boundedLB_nrm D0 D1);
    err := ⋃ (boundedLB_err D0 D1);
    inf := Sets.general_union (is_inf D0 D1);
  |}.

(** 向地址赋值的语义与原先定义基本相同，只是现在需要规定所有变量的地址不被改变，
    而非所有变量的值不被改变。*)

Definition asgn_deref_sem_nrm
             (D1 D2: state -> int64 -> Prop)
             (s1 s2: state): Prop :=
  exists i1 i2,
    D1 s1 i1 /\
    D2 s1 i2 /\
    s1.(mem) i1 <> None /\
    s2.(mem) i1 = Some (Vint i2) /\
    (forall X, s1.(env) X = s2.(env) X) /\
    (forall p, i1 <> p -> s1.(mem) p = s2.(mem) p).

Definition asgn_deref_sem_err
             (D1: state -> int64 -> Prop)
             (s1: state): Prop :=
  exists i1,
    D1 s1 i1 /\
    s1.(mem) i1 = None.

Definition asgn_deref_sem
             (D1 D2: EDenote): CDenote :=
  {|
    nrm := (asgn_deref_sem_nrm D1.(nrm) D2.(nrm));
    err := D1.(err) ∪ D2.(err) ∪
           asgn_deref_sem_err D2.(nrm);
    inf := ∅;
  |}.

(** 变量赋值的行为可以基于此定义。*)




Definition asgn_var_sem
             (X: var_name)
             (D1: EDenote): CDenote :=
  asgn_deref_sem (var_sem_l X) D1.

Check 12.
(* 定义 int64 序列生成函数 *)
(*经过语法变换，在whileD+malloc中的declare语句应该传递两个参数，变量名和需要的地址数*)
Definition declare_sem_malloc (X: var_name) (sz: Z): CDenote :=
{|
    nrm:= fun s1 s2 =>
            forall i, s1.(env) X <> i /\
            exists i': int64, (s2.(env) X = i' /\
            ((s1.(mem) i') <> None)) /\
            forall k: Z, 0 <= k < sz -> ((s1.(mem) (Int64.add i' (Int64.repr k)) = Some Vuninit) /\
            (s2.(mem) (Int64.add i' (Int64.repr k)) = Some (Vint (Int64.repr 0))) /\
            (forall k': Z, k' < (Int64.unsigned i') \/ k' >= (Int64.unsigned i')+sz -> s1.(mem) (Int64.repr k') = s2.(mem) (Int64.repr k')) /\
            (forall X', X' <> X -> s1.(env) X' = s2.(env) X')
            );(**)
    err:= fun s =>exists i, s.(env) X = i;
    inf := ∅;
|}.

Inductive com : Type :=
  | CSkip: com
  | CAsgnVar (x: var_name) (e: expr_malloc): com
  | CAsgnDeref (e1 e2: expr_malloc): com
  | CSeq (c1 c2: com): com
  | CIf (e: expr_malloc) (c1 c2: com): com
  | CWhile (e: expr_malloc) (c: com): com
  | CDeclare (x: var_name) (sz: Z): com.


Fixpoint eval_com (c: com): CDenote :=
  match c with
  | CSkip =>
      skip_sem
  | CAsgnVar X e =>
      asgn_var_sem X (eval_r e)
  | CAsgnDeref e1 e2 =>
      asgn_deref_sem (eval_r e1) (eval_r e2)
  | CSeq c1 c2 =>
      seq_sem (eval_com c1) (eval_com c2)
  | CIf e c1 c2 =>
      if_sem (eval_r e) (eval_com c1) (eval_com c2)
  | CWhile e c1 =>
      while_sem (eval_r e) (eval_com c1)
  | CDeclare X sz =>
      declare_sem_malloc X sz
  end.

End WhileDmalloc.


Import Lang_WhileDU6.
Import WhileDmalloc.
Import EDenote_type.
Import CDenote_type.
Import EDenote.
Import CDenote.

Definition state_type:=Lang_WhileDU6.state.
Definition com_type:=Lang_WhileDU6.com.
Notation "x '.(env')'" := ( Lang_WhileDU6.env x) (at level 1).
Notation "x '.(mem')'" := ( Lang_WhileDU6.mem x) (at level 1).
Notation "x '.(type_env)'" := ( Lang_WhileDU6.type_env x) (at level 1).

Notation "x '.(type_nrm)'" := (EDenote_type.type_nrm x)
  (at level 1, only printing).

Notation "x '.(type_err)'" := (EDenote_type.type_err x)
  (at level 1, only printing).

  Definition Lang_WhileDU6_to_WhileDmalloc_state
  (s1: Lang_WhileDU6.state)
: state :=
{| env := s1.(env');
mem := s1.(mem');
|}.


Fixpoint WhileDU1_to_Malloc_Expr
  (ty_env:Lang_WhileDU6.type_env_state)
            (e1: expr_type)
            : expr_malloc :=  
  match e1 with
  | Lang_WhileDU6.EConst n ty => EConst n
  | Lang_WhileDU6.EVar X ty => EVar X
  | Lang_WhileDU6.EBinop op e1 e2 ty => EBinop op (WhileDU1_to_Malloc_Expr ty_env e1) (WhileDU1_to_Malloc_Expr ty_env e2)
  | Lang_WhileDU6.EUnop op e1 ty => EUnop op (WhileDU1_to_Malloc_Expr ty_env e1)
  | Lang_WhileDU6.EDeref e1 ty => EDeref (WhileDU1_to_Malloc_Expr ty_env e1)
  | Lang_WhileDU6.EAddrOf e1 ty => EAddrOf (WhileDU1_to_Malloc_Expr ty_env e1)
  | Lang_WhileDU6.EStructMember e1 field ty =>
        match expr_type_extract e1 with
        | TStruct struct_name =>
          match  find_field_offset ty_env field (ty_env.(struct_info) struct_name) with
          | Some offset => EDeref(EBinop OPlus (EAddrOf (WhileDU1_to_Malloc_Expr ty_env e1)) (EConst offset))
          | None => EConst 0
          end
        | _ => EConst 0
        end

  | Lang_WhileDU6.EUnionMember e1 field ty =>
        match expr_type_extract e1 with
        | TUnion union_name =>
          match  find_field_offset ty_env field (ty_env.(union_info) union_name) with
          | Some offset => EDeref(EBinop OPlus (EAddrOf (WhileDU1_to_Malloc_Expr ty_env e1)) (EConst 0))
          | None => EConst 0
          end
        | _ => EConst 0
        end

  | Lang_WhileDU6.EPoniter_Struct_Member e1 field ty =>
        match expr_type_extract e1 with
        | TPointer (TStruct struct_name) =>
          match  find_field_offset ty_env field (ty_env.(struct_info) struct_name) with
          | Some offset => EDeref(EBinop OPlus (WhileDU1_to_Malloc_Expr ty_env e1) (EConst offset))
          | None => EConst 0
          end
        | _ => EConst 0
        end

  | Lang_WhileDU6.EPoniter_Union_Member e1 field ty =>
        match expr_type_extract e1 with
        | TPointer (TUnion union_name) =>
          match  find_field_offset ty_env field (ty_env.(union_info) union_name) with
          | Some offset => EDeref(EBinop OPlus (WhileDU1_to_Malloc_Expr ty_env e1) (EConst 0))
          | None => EConst 0
          end
        | _ => EConst 0
        end
  end.
  
  
  Fixpoint Lang_WhileDU6_to_WhileDmalloc_com
            (ty_env:Lang_WhileDU6.type_env_state)
            (c1: com_type)
            : com :=
  match c1 with
  | Lang_WhileDU6.CSkip => CSkip
  | Lang_WhileDU6.CAsgnVar X e1 => CAsgnVar X (WhileDU1_to_Malloc_Expr ty_env e1)
  | Lang_WhileDU6.CAsgnDeref e1 e2 => CAsgnDeref (WhileDU1_to_Malloc_Expr ty_env e1) (WhileDU1_to_Malloc_Expr ty_env e2)
  | Lang_WhileDU6.CSeq c1 c2 => CSeq (Lang_WhileDU6_to_WhileDmalloc_com  ty_env c1) (Lang_WhileDU6_to_WhileDmalloc_com  ty_env c2)
  | Lang_WhileDU6.CIf e1 c1 c2 => CIf (WhileDU1_to_Malloc_Expr ty_env e1) (Lang_WhileDU6_to_WhileDmalloc_com ty_env c1) (Lang_WhileDU6_to_WhileDmalloc_com ty_env c2)
  | Lang_WhileDU6.CWhile e1 c1 => CWhile (WhileDU1_to_Malloc_Expr ty_env e1) (Lang_WhileDU6_to_WhileDmalloc_com ty_env c1)
  | Lang_WhileDU6.CDeclare X ty => CDeclare X (ty_env.(type_size) ty)
  | Lang_WhileDU6.CAsgnBasedOnExp X e1 => CAsgnDeref (EAddrOf (WhileDU1_to_Malloc_Expr ty_env X)) 
                                                      (WhileDU1_to_Malloc_Expr ty_env e1)
  end.


(*Start Refine*)

Definition state_equiv (s1:Lang_WhileDU6.state )(s2: state): Prop :=
  s1.(env') = s2.(env) /\
  s1.(mem') = s2.(mem).

Record erefine_r (e1: expr_malloc ) (ty_env:type_env_state) (e2:expr_type): Prop := {
    nrm_erefine:
      forall (s1:state) (s2: Lang_WhileDU6.state) int64_1,
            (eval_r e1).(nrm) s1 int64_1 /\ 
            (state_equiv s2 s1)  ->
            (Lang_WhileDU6.eval_r e2).(type_nrm) ty_env s2 int64_1\/ 
            (Lang_WhileDU6.eval_r e2).(type_err) ty_env s2;
      
    err_erefine:
      forall (s1:state)(s2: Lang_WhileDU6.state) , 
            (eval_r e1).(err) s1 /\ 
            (state_equiv s2 s1)  ->
            (Lang_WhileDU6.eval_r e2).(type_err) ty_env s2;    
        }.

Lemma var_refine: forall( X : var_name ) (ty:type) (ty_env: type_env_state)  ,
        erefine_r (EVar X ) ty_env (Lang_WhileDU6.EVar X ty).
        Proof.
        intros.
        split.
        +
          intros.
          destruct H.
          destruct H.
          assert(s2.(type_env) X = ty \/ s2.(type_env) X <> ty).
          -
            tauto.
          -
            destruct H1.
            *
              
              destruct ty.
              ++
                left.
                simpl in H.
                sets_unfold in H.
                destruct H.
                unfold Lang_WhileDU6.deref_sem_nrm in H2.
                unfold Lang_WhileDU6.eval_r.
                unfold Lang_WhileDU6.var_sem_r.
                unfold Lang_WhileDU6.deref_sem_r.
                simpl.
                sets_unfold.
                simpl. split.
                +++
                  tauto.
                +++
                  unfold Lang_WhileDU6.deref_sem_nrm.
                  exists x.
                  destruct H0.
                  rewrite H0.
                  rewrite H3.
                  tauto.
              ++
                left.
                simpl in H.
                sets_unfold in H.
                destruct H.
                unfold Lang_WhileDU6.deref_sem_nrm in H2.
                unfold Lang_WhileDU6.eval_r.
                unfold Lang_WhileDU6.var_sem_r.
                unfold Lang_WhileDU6.deref_sem_r.
                simpl.
                sets_unfold.
                simpl. split.
                +++
                  tauto.
                +++
                  unfold Lang_WhileDU6.deref_sem_nrm.
                  exists x.
                  destruct H0.
                  rewrite H0.
                  rewrite H3.
                  tauto.
              ++
                right.
                simpl in H.
                unfold Lang_WhileDU6.eval_r.
                unfold Lang_WhileDU6.False_sem.
                simpl.
                sets_unfold.
                tauto.
              ++
                right.
                simpl in H.
                unfold Lang_WhileDU6.eval_r.
                unfold Lang_WhileDU6.False_sem.
                simpl.
                sets_unfold.
                tauto.
            *
            right.
            unfold Lang_WhileDU6.eval_r.
            unfold Lang_WhileDU6.False_sem.
            destruct ty.
            ++
              simpl.
              sets_unfold.
              left.
              tauto.
            ++
              simpl.
              sets_unfold.
              left.
              tauto.
            ++ 
              simpl.
              sets_unfold.
              
              tauto.
            ++
              simpl.
              sets_unfold.
              tauto.
        +
          intros.
          destruct H.
          assert(s2.(type_env) X = ty \/ s2.(type_env) X <> ty).
          -tauto.
          -
            destruct H1.
            *
              
              destruct ty.
              ++
                unfold Lang_WhileDU6.eval_r.
                unfold Lang_WhileDU6.var_sem_r.
                simpl.
                sets_unfold.
                right.
                right.
                unfold Lang_WhileDU6.deref_sem_err.
                unfold eval_r in H.
                unfold var_sem_r in H.
                unfold deref_sem_r in H.
                simpl in H.
                sets_unfold in H.
                destruct H.
                --
                  tauto.
                --
                  unfold deref_sem_err in H.
                  destruct H.
                  exists x.
                  destruct H0.
                  rewrite H0.
                  rewrite H2.
                  tauto.
              ++
              unfold Lang_WhileDU6.eval_r.
                unfold Lang_WhileDU6.var_sem_r.
                simpl.
                sets_unfold.
                right.
                right.
                unfold Lang_WhileDU6.deref_sem_err.
                unfold eval_r in H.
                unfold var_sem_r in H.
                unfold deref_sem_r in H.
                simpl in H.
                sets_unfold in H.
                destruct H.
                --
                  tauto.
                --
                  unfold deref_sem_err in H.
                  destruct H.
                  exists x.
                  destruct H0.
                  rewrite H0.
                  rewrite H2.
                  tauto.
              ++
                unfold Lang_WhileDU6.eval_r.
                unfold Lang_WhileDU6.False_sem.
                simpl.
                sets_unfold.
                tauto.
              ++
                unfold Lang_WhileDU6.eval_r.
                unfold Lang_WhileDU6.False_sem.
                simpl.
                sets_unfold.
                tauto.
            *
              destruct ty.
              ++
                unfold Lang_WhileDU6.eval_r.
                unfold Lang_WhileDU6.var_sem_r.
                simpl.
                sets_unfold.
                left.
                tauto.
                ++
                unfold Lang_WhileDU6.eval_r.
                unfold Lang_WhileDU6.var_sem_r.
                simpl.
                sets_unfold.
                left.
                tauto.
                ++
                unfold Lang_WhileDU6.eval_r.
                unfold Lang_WhileDU6.var_sem_r.
                simpl.
                sets_unfold.
                
                tauto.
                ++
                unfold Lang_WhileDU6.eval_r.
                unfold Lang_WhileDU6.var_sem_r.
                simpl.
                sets_unfold.
                
                tauto.
            Qed.


            Lemma const_refine: forall( n : Z ) (ty:type) (ty_env: type_env_state)  ,
erefine_r (EConst n ) ty_env (Lang_WhileDU6.EConst n ty).
Proof.
intros.
split.
+
  intros.
  destruct H.
  assert(TInt = ty \/ TInt <> ty) by tauto.
  destruct H1.
  -
    
    destruct ty.
      --
        left.
        simpl in H.
        simpl.
        tauto.
      --
        simpl.
        sets_unfold.
        tauto.
      --
        simpl.
        sets_unfold;
        tauto.
      --
        simpl.
        sets_unfold;
        tauto.
  -
    right.
    simpl.
    destruct ty.
    --
      tauto.
    --
      unfold Lang_WhileDU6.False_sem.
      simpl.
      sets_unfold.
      tauto.
    --
      unfold Lang_WhileDU6.False_sem.
      simpl.
      sets_unfold.
      tauto.
    --
      unfold Lang_WhileDU6.False_sem.
      simpl.
      sets_unfold.
      tauto.
+
  intros.
  destruct H.
  simpl in H.
  simpl.
  destruct ty.
  - 
    simpl.
    tauto.
  -
    unfold Lang_WhileDU6.False_sem.
    simpl.
    sets_unfold.
    tauto.
  -
    unfold Lang_WhileDU6.False_sem.
    simpl.
    sets_unfold.
    tauto.
  -
    unfold Lang_WhileDU6.False_sem.
    simpl.
    sets_unfold.
    tauto.
Qed.


Lemma unop_refine:  forall( op : unop )  (e2: expr_type) (e1:expr_malloc)  (ty:type)  (ty_env: type_env_state),

erefine_r (e1 ) ty_env ( e2 ) ->
(*注释：e1,e2满足精化关系*)
erefine_r  (EUnop op (e1))
            ty_env
            (Lang_WhileDU6.EUnop op e2 ty).
(*尝试证明e1经历两种语言各自的Unop后仍然保持精化关系 *)
Proof.
intros.
split.
+
induction op.
++
intros.
destruct H.
destruct H0.
destruct H. 
destruct H.
pose proof nrm_erefine0 s1 s2 x .
destruct H2;
try tauto.
+++
induction ty eqn:ty1;
simpl;
sets_unfold;
try tauto;
induction (expr_type_extract e2) eqn:e2_type;
simpl;
sets_unfold;
try tauto;
simpl.
- left.
  unfold Lang_WhileDU6.not_sem_nrm.
  exists x.
  tauto.
- left.
  unfold Lang_WhileDU6.not_sem_nrm.
  exists x.
  tauto.
+++
induction ty eqn:ty1;
simpl;
sets_unfold;
try tauto;
induction (expr_type_extract e2) eqn:e2_type;
simpl;
sets_unfold;
try tauto;
simpl.
++
intros.
destruct H.
destruct H0.
simpl in  H. 
pose proof nrm_erefine0 s1 s2 int64_1 .
destruct H1;
try tauto.
+++
induction ty eqn:ty1;
simpl;
sets_unfold;
try tauto;
induction (expr_type_extract e2) eqn:e2_type;
simpl;
sets_unfold;
try tauto.
+++
induction ty eqn:ty1;
simpl;
sets_unfold;
try tauto;
induction (expr_type_extract e2) eqn:e2_type;
simpl;
sets_unfold;
try tauto;
simpl.
+
induction op.
++
intros.
destruct H.
destruct H0.
simpl in  H. 
pose proof err_erefine0 s1 s2  .
assert( (Lang_WhileDU6.eval_r e2).(type_err) ty_env s2) by tauto.
induction ty eqn:ty1;
simpl;
sets_unfold;
try tauto;
induction (expr_type_extract e2) eqn:e2_type;
simpl;
sets_unfold;
try tauto;
simpl.
++
intros.
destruct H.
destruct H0.
simpl in  H. 
pose proof err_erefine0 s1 s2  .
assert( (Lang_WhileDU6.eval_r e2).(type_err) ty_env s2) by tauto.
induction ty eqn:ty1;
simpl;
sets_unfold;
try tauto;
induction (expr_type_extract e2) eqn:e2_type;
simpl;
sets_unfold;
try tauto;
simpl.
Qed.


Lemma dref_refine:  forall  (e2: expr_type) (e1:expr_malloc)  (ty:type)  (ty_env: type_env_state),

erefine_r (e1 ) ty_env ( e2 ) ->
(*注释：e1,e2满足精化关系*)
erefine_r  ( EDeref e1 )
            ty_env
            (Lang_WhileDU6.EDeref  e2 ty).
Proof.
split.
+
intros.
induction ty eqn:ty1;
simpl;
sets_unfold;
try tauto;
induction (expr_type_extract e2) eqn:e2_type;
simpl;
sets_unfold;
try tauto;
simpl.
++
destruct H.
destruct H0.
destruct H.
destruct H.
pose proof nrm_erefine0 s1 s2 x .
destruct H2;
try tauto.
left.
unfold Lang_WhileDU6.deref_sem_nrm.
exists x.
unfold state_equiv in H0.
destruct H0.
rewrite H3.
tauto.
++
destruct H.
destruct H0.
destruct H.
destruct H.
pose proof nrm_erefine0 s1 s2 x .
destruct H2;
try tauto.
left.
unfold Lang_WhileDU6.deref_sem_nrm.
exists x.
unfold state_equiv in H0.
destruct H0.
rewrite H3.
tauto.
+
intros.
induction ty eqn:ty1;
simpl;
sets_unfold;
try tauto;
induction (expr_type_extract e2) eqn :e2_type;
simpl;
sets_unfold;
try tauto;
simpl.
destruct H0.
simpl in H0.
sets_unfold in H0.
++
destruct H0.
+++
left.
destruct H.
pose proof err_erefine0 s1 s2 .
assert( (Lang_WhileDU6.eval_r e2).(type_err) ty_env s2) by tauto.
tauto.
+++
unfold deref_sem_err in H0.
destruct H0.
destruct H.
pose proof nrm_erefine0 s1 s2 x .
destruct H.
try tauto.
++++
right.
unfold Lang_WhileDU6.deref_sem_err.
exists x.
destruct H1.
rewrite H2.
tauto.
++++
left.
tauto.
++
destruct H0.
simpl in H0.
sets_unfold in H0.
destruct H0.
+++
left.
destruct H.
pose proof err_erefine0 s1 s2 .
assert( (Lang_WhileDU6.eval_r e2).(type_err) ty_env s2) by tauto.
tauto.
+++
unfold deref_sem_err in H0.
destruct H0.
destruct H.
pose proof nrm_erefine0 s1 s2 x .
destruct H.
try tauto.
++++
right.
unfold Lang_WhileDU6.deref_sem_err.
exists x.
destruct H1.
rewrite H2.
tauto.
++++
left.
tauto.
Qed.

(*Lemma  arith_sem1_nrm_mono: forall Zfun (e21: expr_type) (e22: expr_type) (e11:expr_malloc) 
(e12: expr_malloc) (ty: type) (ty_env: type_env_state)),
erefine_r (e11 ) ty_env ( e21 ) ->
erefine_r (e12 ) ty_env ( e22 ) ->
erefine_r  (arith_sem1_nrm Zfun e11 e12)
            ty_env
            (Lang_WhileDU6.EBinop Lang_WhileDU6.arith_sem1_nrm Zfun e21 e22 ty).
Proof.
  
Qed.*)


Lemma binop_refine:  forall( op : binop )  (e21: expr_type) (e22: expr_type) (e11:expr_malloc) 
  (e12: expr_malloc) (ty: type) (ty_env: type_env_state),
erefine_r (e11 ) ty_env ( e21 ) ->
erefine_r (e12 ) ty_env ( e22 ) ->
erefine_r  (EBinop op e11 e12)
            ty_env
            (Lang_WhileDU6.EBinop op e21 e22 ty).
Proof.
  intros.
  split.
  + induction op.
    * intros s1 s2 x0 H1.
      destruct H.
      destruct H0.
      destruct H1.
      destruct H.
      destruct H.
      pose proof nrm_erefine0 s1 s2 x.
      destruct H2;
      try tauto.
      - induction ty eqn:ty1;
        simpl;
        sets_unfold;
        try tauto;
        induction (expr_type_extract e21) eqn:e21_type;
        induction (expr_type_extract e22) eqn:e22_type;
        simpl;
        sets_unfold;
        try tauto;
        simpl.
        -- destruct H1.
          --- left.
            unfold Lang_WhileDU6.or_sem_nrm.
            exists x; split; try tauto.
          --- destruct H1.
            destruct H3.
            pose proof nrm_erefine1 s1 s2 x1.
            destruct H4.
            try tauto.
            ---- left.
              unfold Lang_WhileDU6.or_sem_nrm.
              exists x; split; try tauto.
              right; split; try tauto.
              exists x1; split; try tauto.
            ---- right.
              right.
              unfold Lang_WhileDU6.or_sem_err.
              exists x; tauto.
        -- destruct H1.
          --- left.
            unfold Lang_WhileDU6.or_sem_nrm.
            exists x; split; try tauto.
          --- destruct H1.
            destruct H3.
            pose proof nrm_erefine1 s1 s2 x1.
            destruct H4.
            try tauto.
            ---- left.
              unfold Lang_WhileDU6.or_sem_nrm.
              exists x; split; try tauto.
              right; split; try tauto.
              exists x1; split; try tauto.
            ---- right.
              right.
              unfold Lang_WhileDU6.or_sem_err.
              exists x; tauto.
        -- destruct H1.
          --- left.
            unfold Lang_WhileDU6.or_sem_nrm.
            exists x; split; try tauto.
          --- destruct H1.
            destruct H3.
            pose proof nrm_erefine1 s1 s2 x1.
            destruct H4.
            try tauto.
            ---- left.
              unfold Lang_WhileDU6.or_sem_nrm.
              exists x; split; try tauto.
              right; split; try tauto.
              exists x1; split; try tauto.
            ---- right.
              right.
              unfold Lang_WhileDU6.or_sem_err.
              exists x; tauto.
          -- destruct H1.
            --- left.
              unfold Lang_WhileDU6.or_sem_nrm.
              exists x; split; try tauto.
            --- destruct H1.
              destruct H3.
              pose proof nrm_erefine1 s1 s2 x1.
              destruct H4.
              try tauto.
              ---- left.
                unfold Lang_WhileDU6.or_sem_nrm.
                exists x; split; try tauto.
                right; split; try tauto.
                exists x1; split; try tauto.
              ---- right.
                right.
                unfold Lang_WhileDU6.or_sem_err.
                exists x; tauto.
        - induction ty eqn:ty1;
          simpl;
          sets_unfold;
          try tauto;
          induction (expr_type_extract e21) eqn:e21_type;
          induction (expr_type_extract e22) eqn:e22_type;
          simpl;
          sets_unfold;
          try tauto;
          simpl.
      * 
          
          exists x.
          split; try tauto.
          try tauto.
          right.
          destruct H1.
          split; try tauto.
          destruct H3.
          destruct H3.
          destruct H4.
          try tauto.
          --- exists x1; split; try tauto.
          --- exists x1; split; try tauto.


      
        -- destruct H3.
          --- unfold SC_or_compute_nrm in H1.
Qed.