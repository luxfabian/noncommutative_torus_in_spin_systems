#!/usr/bin/env wolframscript
(* ::Package:: *)

(* ::Title:: *)
(*Calculation of Chern numbers*)


(* ::Subtitle:: *)
(*K-theory expansion*)


(* ::Section:: *)
(*General Definitions*)


(* ::Text:: *)
(*Construction of the flux matrix. v is a vector of generators. th maps the generators to a number*)


(* ::Input::Initialization:: *)
Theta[v_]:= Array[ th[ v[[#1]] , v[[#2]] ]  & , {Length[v], Length[v]} ] 


(* ::Text:: *)
(*Array of all possible complements where J is the list of all subsets of the set of generators*)


(* ::Input::Initialization:: *)
JJ:= Array[ Complement[J[[#1]], J[[#2]] ]&, {Length[J], Length[J] } ];
\[CapitalPhi][i_,j_]:= Theta[JJ[[i,j]]];


(* ::Text:: *)
(*From these, the Chern numbers can be calculated*)


(* ::Input::Initialization:: *)
ChernElem[i_,j_]:=(

len = Length[ JJ[[i,j]] ];

If[ J[[i]] == J[[j]] , Return[1] ];
If[ len==0, Return[0  ] ]; 

(*Pfaffian*)
Return[ Sqrt[ Det[ \[CapitalPhi][i,j]]]]; 

)


(* ::Text:: *)
(*The j-th Chern number*)


Chern[j_]:={J[[j]],  Sum[ n[J[[i]]]ChernElem[i,j],{i,1,Length[J] } ]} 


(* ::Text:: *)
(*Generate table*)


ChernTable := Chern[#]&/@Range[1,Length[J] ] // Simplify


(* ::Section:: *)
(*Helicoids*)


(* ::Input::Initialization:: *)
r = 2;
d = 1;
vars= ( t[#]&/@ Range[ d] ) ~ Join ~ ( u[#]&/@ Range[ r] );

svars=DeleteCases[ Subsets[vars], {}] ;
J = Join[{{}},Select[ svars, EvenQ[Length[#]] &]];

\[Theta][1] = { \[Theta]1};
\[Theta][2] = { \[Theta]2};
$Assumptions={\[Theta]1>0,\[Theta]2>0};

tmat = \[Theta][#]&/@Range[r];

th[t[i_], u[j_] ]:=tmat  [[j,i]];
th[u[j_], t[i_] ]:= -tmat  [[j,i]]
th[u[i_], u[j_] ]:= 0
th[t[i_], t[j_] ]:= 0


ChernTable // Expand //Grid


 Join[{{"J'","\!\(\*SubscriptBox[\(Ch\), \(J'\)]\)"}}, Expand[ChernTable ] ]/.{t[x_]-> Subscript[tau, x], u[x_] -> Subscript[u, x]}/.{ n[x_]-> Subscript[n, x],\[Theta]-> Subscript[theta, 1] ,\[Theta]1-> Subscript[theta, 1],\[Theta]2-> Subscript[theta, 2]}  //Grid // TeXForm


(* ::Section:: *)
(*2Q Triangular Skyrmion Lattice*)


(* ::Input::Initialization:: *)
r =2;
d = 2;
vars= ( t[#]&/@ Range[ d] ) ~ Join ~ ( u[#]&/@ Range[ r] );

svars=DeleteCases[ Subsets[vars], {}] ;
J = Join[{{}},Select[ svars, EvenQ[Length[#]] &]];

tmat = \[Theta]{{0,1},{1,0}};
$Assumptions={\[Theta]>0};

th[t[i_], u[j_] ]:=tmat  [[j,i]];
th[u[j_], t[i_] ]:= -tmat  [[j,i]]
th[u[i_], u[j_] ]:= 0
th[t[i_], t[j_] ]:= 0


ct = ChernTable// Expand;


ct // MatrixForm


eqs = Table[ Ch[ct[[i]][[1]]] == ct[[i]] [[2]] , {i, 1, Length[J]} ] ;
X =  Table[ n[J[[i]]]  , {i, 1, Length[J]} ] ;
Y =  Table[ Ch[J[[i]]]  , {i, 1, Length[J]} ] ;


sol = Solve[eqs,X][[1]] ;


(n[{t[1],u[1]}]+ n[{t[1],u[2]}] ) /.sol //FullSimplify


ChernTable // Expand //Grid


 Join[{{"J'","\!\(\*SubscriptBox[\(Ch\), \(J'\)]\)"}}, Expand[ChernTable ] ]/.{t[x_]-> Subscript[tau, x], u[x_] -> Subscript[u, x]}/.{ n[x_]-> Subscript[n, x],\[Theta]-> Subscript[theta, 1] }  //Grid // TeXForm


(* ::Section:: *)
(*3Q Triangular Skyrmion Lattice*)


(* ::Input::Initialization:: *)
r = 3;
d = 2;
vars= ( t[#]&/@ Range[ d] ) ~ Join ~ ( u[#]&/@ Range[ r] );

svars=DeleteCases[ Subsets[vars], {}] ;
J = Join[{{}},Select[ svars, EvenQ[Length[#]] &]];

tmat = \[Theta]{{0,1},{1,0},{-1,-1}};
$Assumptions={\[Theta]>0};

th[t[i_], u[j_] ]:=tmat  [[j,i]];
th[u[j_], t[i_] ]:= -tmat  [[j,i]];
th[u[i_], u[j_] ]:= 0
th[t[i_], t[j_] ]:= 0


ct = ChernTable// Expand;


eqs = Table[ Ch[ct[[i]][[1]]] == ct[[i]] [[2]] , {i, 1, Length[J]} ] ;
X =  Table[ n[J[[i]]]  , {i, 1, Length[J]} ] ;
Y =  Table[ Ch[J[[i]]]  , {i, 1, Length[J]} ] ;


sol = Solve[eqs,X][[1]] ;


n0 = ( n[J[[1]]] /.sol // Expand ) //ComplexExpand;


nn = X/.sol//Simplify//Expand; 


 Join[{{"J'","\!\(\*SubscriptBox[\(Ch\), \(J'\)]\)"}}, Expand[ChernTable ] ]/.{t[x_]-> Subscript[tau, x], u[x_] -> Subscript[u, x]}/.{ n[x_]-> Subscript[n, x],\[Theta]-> Subscript[theta, 1] }  //Grid // TeXForm


(* ::Section:: *)
(*3Q Square Hedgehog Lattice*)


(* ::Input::Initialization:: *)
r = 3;
d = 3;
vars= ( t[#]&/@ Range[ d] ) ~ Join ~ ( u[#]&/@ Range[ r] );

svars=DeleteCases[ Subsets[vars], {}] ;
J = Join[{{}},Select[ svars, EvenQ[Length[#]] &]];

tmat = \[Theta] IdentityMatrix[3];
$Assumptions={\[Theta]>0};

th[t[i_], u[j_] ]:=tmat  [[j,i]];
th[u[j_], t[i_] ]:=-tmat  [[j,i]]
th[u[i_], u[j_] ]:= 0
th[t[i_], t[j_] ]:= 0


Chern[1] //Simplify // Expand


 Join[{{"J'","\!\(\*SubscriptBox[\(Ch\), \(J'\)]\)"}}, Expand[ChernTable ] ]/.{t[x_]-> Subscript[tau, x], u[x_] -> Subscript[u, x]}/.{ n[x_]-> Subscript[n, x],\[Theta]-> Subscript[theta, 1] }  //Grid // TeXForm


(* ::Section:: *)
(*Effective Model 2D*)


(* ::Input::Initialization:: *)
r =0;
d = 2;
vars= ( t[#]&/@ Range[ d] ) ~ Join ~ ( u[#]&/@ Range[ r] );

svars=DeleteCases[ Subsets[vars], {}] ;
J = Join[{{}},Select[ svars, EvenQ[Length[#]] &]];

chi[x_,y_]:=0
chi[1,2] :=\[Theta]^2
chi[2,1] :=-\[Theta]^2


th[t[i_], u[j_] ]:=0
th[u[j_], t[i_] ]:=0
th[u[i_], u[j_] ]:= 0
th[t[i_], t[j_] ]:=chi[i,j]


ChernTable // Expand //Grid


 Join[{{"J'","\!\(\*SubscriptBox[\(Ch\), \(J'\)]\)"}}, Expand[ChernTable ] ]/.{t[x_]-> Subscript[s, x], u[x_] -> Subscript[u, x]}/.{ n[x_]-> Subscript[n, x],\[Theta]-> Subscript[theta, 1] ,\[Theta]1-> Subscript[theta, 1],\[Theta]2-> Subscript[theta, 2]}  //Grid // TeXForm


(* ::Section:: *)
(*1Q 3D*)


(* ::Input::Initialization:: *)
r = 1;
d = 3;
vars= ( t[#]&/@ Range[ d] ) ~ Join ~ ( u[#]&/@ Range[ r] );

svars=DeleteCases[ Subsets[vars], {}] ;
J = Join[{{}},Select[ svars, EvenQ[Length[#]] &]];

\[Theta][1] = \[Theta]1{0,0,1 };
$Assumptions={\[Theta]1>0,\[Theta]2>0};

tmat = \[Theta][#]&/@Range[r];

th[t[i_], u[j_] ]:=tmat  [[j,i]];
th[u[j_], t[i_] ]:= -tmat  [[j,i]]
th[u[i_], u[j_] ]:= 0
th[t[i_], t[j_] ]:= 0


ChernTable // Expand //Grid
