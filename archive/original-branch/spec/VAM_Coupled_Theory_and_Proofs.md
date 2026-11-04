---
title: "The Variational Action Machine"
subtitle: "Coupled Markov–Dirac–Zurek dynamics, bit-level ambiguity, Feynman–Newton inference, and structural retention"
date: "Mathematical development from the original paper and subsequent author clarifications"
---

# 1. Scope, source, and the principal result

This report develops *Foundational Principles — The Variational Action Machine & Zero-Memory Architecture* as a standalone proposal. The author's subsequent explanations are part of the specification. In particular: Dirac dynamics are to be used **inside the open-system setting of Zurek's decoherence and einselection**, not merely as an analogy for local propagation; Markov structure determines present-state sufficiency and boundary-mediated interaction; incoming bits need not already constitute a complete meaningful unit; and every emitted configuration re-enters the same evolving organization before the next external configuration is processed.

The original paper supplies the mechanisms. The work here is to connect them mathematically, not replace them with a conventional path planner, an external correctness supervisor, a neural encoder, or a history-retrieval architecture.

**Principal result.** A well-defined event loop can be constructed from joint vector–weight density states, Dirac-generated interactions, boundary-conditioned state changes, and idempotent canalization. Within explicit examples, it supports unresolved input boundaries, ordered compositional change, reciprocal vector–weight influence, coherent interference, observable output, and immediate output-driven reconfiguration. The finite-dimensional event-loop theorem proves that these operations compose into a normalized, positive, present-state Markov process. Separate derivations establish the classical stationary-action limit and exact boundary/canalization conditions.

This proves mathematical compatibility and several concrete mechanisms. It does not, by itself, prove textbook understanding, a uniquely correct answer to every question, or a general computational speedup. Those conclusions would require showing how the evolving interaction structure acquires the appropriate meaning-sensitive dynamics. The distinction is between a specified mechanism and a theorem about what that mechanism learns—not between an existing idea and a supposedly absent architectural component.

## 1.1 What is preserved from the source

| Source mechanism | Mathematical treatment in this report |
|---|---|
| Immutable universal bit substrate, Part 10 | One binary input algebra; incomplete interpretation remains in the current density state. |
| State-as-memory, Part 1 | A controlled Markov event process; no transcript is required for its internal update. |
| Density matrices of vectors and weights, Parts 4, 7, 9 | A joint positive density operator, retaining correlations rather than multiplying unrelated marginals. |
| Markov blanket, Parts 5–6 | Both a conditional-independence condition and a generator-level boundary condition. |
| Dirac inside Zurek, Part 7 and author clarification | A Dirac system–environment Hamiltonian, reduced density evolution, and boundary einselection. |
| Feynman–Newton macro/micro process, Part 2 | Exact coherent path composition and its stationary-action limit, without replacing phases by costs. |
| Token metamorphosis, Part 3 | A measured output immediately drives another interaction of the same machine. |
| Canalization and compiled lock, master loop | A dynamical projection and a proved condition for preserving future observable behavior. |
| Hierarchical compaction, Part 9 | Joint correlations, isometric nesting, and removal of dynamically irrelevant degrees of freedom. |
| Forward plasticity, Part 3 | Reciprocal Hamiltonian coupling and a derived dissipative correlation-to-weight example. |

Throughout, **source statement** means a mechanism or claim actually in the supplied paper or clarification. **Derived result** means a consequence proved from stated equations. **Chosen realization** means a specific interaction, encoding, or parameter value supplied here to make a calculation determinate. A chosen realization is not attributed to the author as the unique intended law.

## 1.2 Notation and mathematical setting

$R$ denotes the density operator of the retained computational organization. It includes vector-bearing degrees of freedom $V$, weights or relational controls $W$, and any unresolved boundary degrees of freedom $B$. A discarded or incoming environmental factor is denoted $E$. The original paper's instantaneous state $S$ can contain a representation of $R$ together with any classical structural coordinates needed to determine the next interaction.

A density operator satisfies

$$
R=R^\dagger,\qquad R\succeq0,\qquad \operatorname{Tr}R=1.
$$

A completely positive map preserves positivity even when the system is part of a larger correlated system. A trace-preserving such map describes an unconditioned physical update. A trace-nonincreasing map describes one possible observed outcome, before normalizing its probability.

Finite matrices are used for the explicit proofs and numerical examples. This is not an architectural prohibition on growth or continuous fields. Infinite-dimensional versions additionally require operator domains, self-adjointness, and convergence conditions. No numerical coefficient is inferred from a physical analogy alone.

# 2. Bits, incomplete configurations, and differential composition

## 2.1 Universal substrate does not require presegmented words

The fixed base is the binary alphabet and its elementary operations. Its constancy does not mean that every input value is fixed. Text, images, and audio can use the same binary substrate while imposing different correlations among incoming bits. A particular bit pattern is unique as a finite string; its interpretation and whether it finishes a meaningful unit depend on the current encoding and context. This is exactly where the author's proposed density representation is useful.

A mathematical reference space for arbitrary finite strings is

$$
\mathcal H_{\mathrm{strings}}
=\bigoplus_{\ell=0}^{\infty}(\mathbb C^2)^{\otimes\ell}.
$$

This is a reference space of possibilities, not an instruction to store every historical string. The actual retained state may instead represent equivalence classes of partial interpretations and their relations. The next bit acts on that state immediately; the machine need not wait for a supplied tokenizer to announce that a word is complete.

Let $P_{\mathrm{complete}}$ select the currently represented completed interpretations. Then

$$
p_{\mathrm{complete}}=\operatorname{Tr}(P_{\mathrm{complete}}R).
$$

Completeness is thus an observable of the present organization, not a compulsory hard decision imposed before processing. The interpretation sectors and their couplings may themselves develop. A numerical example needs a specific set of sectors, but the general construction does not identify them with a fixed list of language tokens or modalities.

## 2.2 Exact example: a bit can complete one interpretation and begin another

**Chosen realization.** Consider three possible encoded units, sampled independently with equal probability when a new unit starts:

$$
A=0,\qquad B=01,\qquad C=10.
$$

There are no delimiters. These codes are intentionally not prefix-free. The three current structural states are $r$ (a unit has just ended), $b$ (the first bit of $B$ has arrived), and $c$ (the first bit of $C$ has arrived). They are not a saved transcript.

For the diagonal density, write its population vector as $z=(z_r,z_b,z_c)^T$. The unnormalized next-bit maps are

$$
M_0=\begin{pmatrix}1/3&0&1\\1/3&0&0\\0&0&0\end{pmatrix},
\qquad
M_1=\begin{pmatrix}0&1&0\\0&0&0\\1/3&0&0\end{pmatrix}.
$$

Given an observed bit $u$, compute

$$
p(u)=\mathbf 1^T M_u z,\qquad z^{\mathrm{new}}=\frac{M_u z}{p(u)}.
$$

Every column of $M_0+M_1$ sums to one, so the next-bit probabilities normalize. Starting from $z=(1,0,0)^T$ gives the complete calculation:

| Received bits | Current populations $(z_r,z_b,z_c)$ | Probability of latest bit | Probability a unit ends here |
|---|---|---:|---:|
| $0$ | $(1/2,1/2,0)$ | $2/3$ | $1/2$ |
| $01$ | $(3/4,0,1/4)$ | $2/3$ | $3/4$ |
| $010$ | $(2/3,1/3,0)$ | $3/4$ | $2/3$ |

The probability of the full observed string $010$ in this specified generative example is

$$
\frac23\frac23\frac34=\frac13.
$$

At the final step, $AC$ and $BA$ are complete parses, while $B$ followed by the beginning of another $B$ is incomplete. The three possibilities each contribute probability $1/9$ before normalization. The state correctly retains the unresolved distinction instead of inventing a delimiter.

### Density-operator realization and proof

The zero-bit map has Kraus operators

$$
K_{0,A}=\frac{\left|r\right\rangle\left\langle r\right|}{\sqrt3},\quad
K_{0,B}=\frac{\left|b\right\rangle\left\langle r\right|}{\sqrt3},\quad
K_{0,C}=\left|r\right\rangle\left\langle c\right|.
$$

The one-bit map has

$$
K_{1,C}=\frac{\left|c\right\rangle\left\langle r\right|}{\sqrt3},\qquad
K_{1,B}=\left|r\right\rangle\left\langle b\right|.
$$

Set $\mathcal B_u(R)=\sum_a K_{u,a}RK_{u,a}^\dagger$. Direct calculation gives

$$
\sum_{u,a}K_{u,a}^\dagger K_{u,a}=I.
$$

Consequently, the maps preserve positivity and jointly preserve trace. Normalizing $\mathcal B_u(R)$ reproduces the table. This is a complete instance of uncertain input completion inside the density dynamics. It does not assume that these three example codes, or their equal prior, are supplied to a general-purpose machine.

## 2.3 A bit stream changes an equation through noncommuting actions

Let $G_0$ and $G_1$ be the current generators activated by the two bit values. The generator can act on vectors, relational controls, and boundary variables jointly. For a short duration $\delta$, one bit acts through $e^{\delta G_u}$.

For successive bits $u_1,u_2$,

$$
v^{\mathrm{new}}=e^{\delta G_{u_2}}e^{\delta G_{u_1}}v.
$$

The Baker–Campbell–Hausdorff expansion, where the series converges, is

$$
\log(e^{\delta G_1}e^{\delta G_0})
=\delta(G_1+G_0)+\frac{\delta^2}{2}[G_1,G_0]+O(\delta^3).
$$

The commutator is the first order-sensitive contribution. Thus a composed configuration can be represented by a changed effective differential generator rather than by replaying its words. The exact event process uses the ordered product; it does not require truncating this expansion or assuming every composition has a short closed form.

**Exact two-component calculation.** Let

$$
U_0=e^{-i\pi Z/4},\qquad U_1=e^{-i\pi X/4},
$$

where $X,Y,Z$ are the Pauli matrices, and start from $\left|0\right\rangle$. These are mass-axis and kinetic-axis pulses in a two-component Dirac-sector illustration. They are not a semantic encoding of arbitrary language.

| Input order | Final Bloch vector $(\left\langle X\right\rangle,\left\langle Y\right\rangle,\left\langle Z\right\rangle)$ | Probability of $X=+1$ |
|---|---|---:|
| $01$ | $(0,-1,0)$ | $1/2$ |
| $10$ | $(1,0,0)$ | $1$ |

For example, $U_1U_0\left|0\right\rangle$ is, up to an overall phase, $(\left|0\right\rangle-i\left|1\right\rangle)/\sqrt2$, whereas $U_0U_1\left|0\right\rangle$ is $(\left|0\right\rangle+\left|1\right\rangle)/\sqrt2$. Order survives in the present state although no two-item history buffer is used.

# 3. A density structure that really includes vectors and weights

The paper's density tensor should be interpreted using its strongest formulation: Part 9 explicitly includes relational and cross-layer correlations. Restricting it to a product of independent local densities would discard information the author intends to retain.

## 3.1 Joint representation

Let $\left|w\right\rangle$ label orthogonal relational-control configurations and $\left|v_w\right\rangle$ be normalized information vectors conditional on those configurations. One valid joint state is

$$
\left|\Psi\right\rangle=\sum_w\sqrt{p_w}\,\left|v_w\right\rangle\otimes\left|w\right\rangle,
\qquad R_{VW}=\left|\Psi\right\rangle\left\langle\Psi\right|.
$$

Its reduced vector density is

$$
R_V=\sum_w p_w\left|v_w\right\rangle\left\langle v_w\right|,
$$

while the weight density has entries

$$
(R_W)_{ww'}=\sqrt{p_wp_{w'}}\left\langle v_{w'},v_w\right\rangle.
$$

This records both the distributions and the relationships. For a general mixed state, use $R=AA^\dagger/\operatorname{Tr}(AA^\dagger)$, or a mixture of such joint pure states. Positivity follows because $x^\dagger AA^\dagger x=\|A^\dagger x\|^2\ge0$.

Weights need not be nonnegative merely because probabilities are. A signed interaction weight is an observable such as $\widehat W=\sum_w w\left|w\right\rangle\left\langle w\right|$, whose eigenvalues can be positive or negative. The density over those possible weights remains positive.

## 3.2 Mixing a bit-derived matrix with a density

For a bit-derived operator $X_u$ acting on a currently represented space, the update

$$
R^{\mathrm{new}}=\frac{X_uRX_u^\dagger}{\operatorname{Tr}(X_uRX_u^\dagger)}
$$

is positive whenever the denominator is nonzero. A physical observation instrument additionally requires the relevant Kraus completeness condition. By contrast, merely writing $R_1R_2$ is not a general way of mixing densities: it can fail to be Hermitian when the factors do not commute. The congruence construction supplies the missing algebra without changing the proposed mechanism.

## 3.3 Exact reciprocal influence from one interaction Hamiltonian

Take

$$
\mathbb H=H_{D,0}\otimes I+I\otimes H_W+\sum_a D_a\otimes\widehat W_a.
$$

Here $H_{D,0}$ is a Dirac generator on information-bearing modes, $H_W$ evolves the relational degrees of freedom, and the last term couples them. All terms are self-adjoint on their declared domain. In a finite discretization, all are Hermitian matrices.

The joint evolution is $\dot R=-i[\mathbb H,R]/\hbar$. For observables $A$ of vectors and $B$ of weights,

$$
\frac{d}{dt}\left\langle A\right\rangle
=\frac{i}{\hbar}\left\langle[H_{D,0},A]\right\rangle
+\frac{i}{\hbar}\sum_a\left\langle[D_a,A]\otimes\widehat W_a\right\rangle,
$$

$$
\frac{d}{dt}\left\langle B\right\rangle
=\frac{i}{\hbar}\left\langle[H_W,B]\right\rangle
+\frac{i}{\hbar}\sum_a\left\langle D_a\otimes[\widehat W_a,B]\right\rangle.
$$

**Derived result.** The same interaction generates both directions of influence. No independently invented reverse-update function is necessary. The instantaneous expectation governing the vector generally includes joint correlations, not just mean scalar weights.

For a product state at a particular instant, the vector's instantaneous effective Hamiltonian is $H_{D,0}+\sum_a\left\langle\widehat W_a\right\rangle D_a$. Once correlations develop, replacing the joint state by that mean-field description is an additional approximation. Section 9 calculates its error explicitly.

## 3.4 A positive density-derived geometry

The paper also proposes that density changes the landscape's geometry. A concrete positive construction is

$$
g_{ab}(R)=g_0\delta_{ab}+\frac12\operatorname{Tr}\!\left(R\{A_a,A_b\}\right),\qquad g_0>0,
$$

for Hermitian observables $A_a$. For every real vector $v$,

$$
v^Tg(R)v=g_0\|v\|^2+
\operatorname{Tr}\!\left[R\left(\sum_a v_aA_a\right)^2\right]\ge g_0\|v\|^2.
$$

Thus the metric is strictly positive and changes when the density changes. This is one possible realization of the paper's density-to-geometry statement, not an identification of a density matrix with a spacetime metric. Feeding it into a curved-space Dirac operator additionally requires a coordinate manifold, a vielbein and spin connection. The joint-Hamiltonian calculations above do not depend on making those extra choices.

# 4. Markov structure: present state, communication, and boundaries

The original paper assigns Markov mathematics two jobs. Both can be formalized, but they are distinct mathematical statements.

## 4.1 Present-state sufficiency

For the complete retained state $S_n$ and the next supplied bit $u$, a specified update kernel satisfies

$$
\Pr(S_{n+1},\tau\mid S_n,u,\text{earlier events})
=\Pr(S_{n+1},\tau\mid S_n,u).
$$

If several earlier streams produce the same complete present configuration, the machine has the same future internal behavior under the same subsequent inputs. This gives the intended zero-history execution rule.

It does not follow that an arbitrarily reduced marginal contains everything needed to predict an environment that later returns to interact. Correlations that can still matter belong in the current joint state or its boundary representation. Retaining them is retaining what the system is, not reintroducing a chronological logbook.

## 4.2 Which parts directly influence which others

For an internal region $I$, blanket $B$, and exterior $E$, realize boundary mediation by

$$
\mathbb H=H_{IB}\otimes I_E+I_I\otimes H_{BE},
$$

with no direct $I$–$E$ interaction term. For an internal observable $O_I$,

$$
[H_{BE},O_I\otimes I_B\otimes I_E]=0.
$$

Therefore

$$
\frac{d}{dt}\left\langle O_I\right\rangle
=\frac{i}{\hbar}\operatorname{Tr}_{IB}
\left(R_{IB}[H_{IB},O_I\otimes I_B]\right).
$$

**Proof.** Insert the Hamiltonian decomposition into the Heisenberg equation. The $BE$ commutator vanishes because its factors act outside $I$. Trace out $E$ in the remaining term. The same conclusion holds for a Lindblad generator $\mathcal L_{IB}+\mathcal L_{BE}$ with trace-preserving local dissipators, since $\mathcal L_{BE}^\dagger(O_I)=0$.

This is an exact statement of boundary-mediated immediate influence. It allows $E$ to affect $I$ subsequently through $B$. It does not require disconnected information to be permanently unable to communicate.

The division can change when the current interaction structure changes. An interaction weight with zero *mean* is not automatically an absent connection: correlations and weight variance can still contribute. Absence is determined by the actual interaction acting on the currently accessible support, not by a single average.

## 4.3 The statistical blanket law

The source equation is

$$
P(I\mid B,E)=P(I\mid B).
$$

A sufficient quantum realization, expressed in a classical pointer basis of the blanket, is

$$
R_{IBE}=\sum_z p_z\,R_I^{(z)}\otimes\left|z\right\rangle\left\langle z\right|_B\otimes R_E^{(z)}.
$$

Conditioning on the complete pointer label $z$ leaves an internal–external product. This proves the source's conditional-independence statement for local interior and exterior measurement outcomes conditioned on that label. Coarsening or measuring the blanket in a different basis need not preserve the same screening property.

The more general exact quantum Markov form permits a decomposition

$$
\mathcal H_B=\bigoplus_z\mathcal H_{B_L^{(z)}}\otimes\mathcal H_{B_R^{(z)}},
$$

$$
R_{IBE}=\bigoplus_z p_z\,
R_{I B_L^{(z)}}\otimes R_{B_R^{(z)}E}.
$$

This is equivalent to vanishing quantum conditional mutual information in finite dimensions, by the theorem of Hayden, Jozsa, Petz and Winter [R3]. It permits entanglement on each side of the blanket; it does not force every local state to be an independent classical probability table.

A useful quantitative diagnostic is

$$
I(I:E\mid B)=S(IB)+S(BE)-S(B)-S(IBE).
$$

When it is nonzero, the proposed statistical boundary does not fully screen the correlations. The source's hierarchy can retain the necessary joint state or use a larger boundary. Small conditional mutual information alone is not an exact equality or a derived bound on dynamical error.

## 4.4 Why the two conditions cannot be conflated

Consider three classical bits with $E$ uniformly random and initially $B=I=0$. Execute, in order, $B\leftarrow B\oplus E$, $I\leftarrow I\oplus B$, and $B\leftarrow B\oplus E$. Every interaction goes through $B$. The final state is $B=0$, $I=E$.

There is no direct interior–exterior interaction, but at the end $I(I:E\mid B)=1$ bit. The blanket has relayed a correlation that its present single bit does not retain. This example does not invalidate the proposed boundary principle; it establishes why a faithful realization must match both its state representation and its interaction law to that principle.

## 4.5 Developing boundaries from the same density–weight state

There is a direct way to let the boundary organization change without adding an external graph-rewriting mechanism. Let a relational weight register contain an activity observable $N_e$ for each potential local interaction $h_e$. For a binary activity register, $N_e$ has eigenvalues $0$ and $1$. Use

$$
H=H_W+\sum_e N_e\otimes h_e.
$$

The set of supports of the $h_e$ is merely a description of which factors each term acts upon. In a pointer configuration $n$ of the relational registers, the instantaneous active interaction set is the set of $e$ with $n_e=1$. The boundary of a chosen internal region consists of exterior factors appearing in one of those active terms, together with any required relational-control factors. Terms in $H_W$ must also be included when constructing the complete boundary.

For three vector factors, consider the two interaction terms

$$
H_{\mathrm{int}}=N_{12}\otimes X_1X_2+N_{23}\otimes X_2X_3.
$$

Then

$$
\dot{\left\langle Z_1\right\rangle}=2\left\langle N_{12}Y_1X_2\right\rangle.
$$

If the weight density is supported on $N_{12}=0$, the first interaction is inactive on that support. If input or output re-entry changes the weight state to $N_{12}=1$, it becomes active. The immediate communication structure has changed through density dynamics itself. If $H_W$ can change $N_{12}$, inactivity at one instant is not a permanent hard constraint.

A density mixture over $n$ describes uncertainty about the currently realized interaction configuration. Coherence between $n$ sectors can also be retained until the relevant boundary interaction distinguishes them. It is then inaccurate to assign one definite classical boundary too early; the joint density or the union of the still-accessible interaction supports is the correct retained description.

This construction ties the author's evolving weights, density uncertainty, Zurek pointer selection and Markov boundaries together. The statistical conditional-independence condition still has to hold for any claimed statistical blanket; changing interaction support does not make it automatic.

# 5. An exact whirlpool-like system with a Markov blanket

The paper connects a spiral sink to a boundary isolating internal dynamics. The following construction proves that these two features can coexist in one model, instead of relying only on an analogy.

Let $X=(x,y,b,e)^T$, with internal variables $(x,y)$, blanket $b$, and exterior $e$. Define a Gaussian stationary density with precision matrix

$$
J=\begin{pmatrix}
2&0&-1&0\\
0&2&0&0\\
-1&0&3&-1\\
0&0&-1&2
\end{pmatrix}.
$$

Its leading principal minors are $2,4,10,16$, so $J$ is positive definite. Let

$$
Q=\begin{pmatrix}
0&-1&0&0\\1&0&0&0\\0&0&0&0\\0&0&0&0
\end{pmatrix},\qquad
A=(Q-I)J.
$$

Use the stochastic differential equation

$$
dX=AX\,dt+\sqrt2\,dW_t.
$$

The stationary covariance is $\Sigma=J^{-1}$, because

$$
A\Sigma+\Sigma A^T+2I=(Q-I)+(Q^T-I)+2I=0.
$$

Since $Q$ is antisymmetric, the drift is stable: it is similar to a matrix with strictly negative-definite symmetric part. The invariant Gaussian distribution is therefore well defined.

The drift matrix evaluates to

$$
A=\begin{pmatrix}
-2&-2&1&0\\
2&-2&-1&0\\
1&0&-3&1\\
0&0&1&-2
\end{pmatrix}.
$$

The first two rows contain no $e$ term: outside influence reaches the interior only through $b$. Meanwhile, conditioning the Gaussian on $b,e$ gives

$$
(x,y)\mid b,e\sim
\mathcal N\!\left(\begin{pmatrix}b/2\\0\end{pmatrix},\frac12I_2\right),
$$

which does not depend on $e$. Thus the stationary conditional-independence blanket and the dynamical boundary both hold exactly.

With $b$ held fixed, the deterministic internal drift around $(b/2,0)$ is

$$
\dot z=\begin{pmatrix}-2&-2\\2&-2\end{pmatrix}z.
$$

Its eigenvalues are $-2\pm2i$, its radius decays as $e^{-2t}$, its angle increases by $2t$, and its planar vorticity is $4$. At $t=\log2$, the radius is one quarter of its initial value.

**Scope of the result.** This is a constructed stochastic spiral system with a proved Markov blanket. It is not a proof that every natural vortex has an exact finite-dimensional Markov blanket, nor an assertion that this Gaussian model must be installed in the computational machine. It makes the source's proposed connection mathematically concrete.

The black-hole comparison has a different status. Stationary black-hole uniqueness results characterize certain exterior solution families under specified field equations and regularity hypotheses [R7]. They do not prove that arbitrary interior information is recoverable from three numbers. The useful common abstraction is an equivalence of internal realizations with respect to declared exterior observables. Canalization below gives that abstraction an explicit computational form without identifying gravity, fluid motion, and quantum reduction as the same physical law.

# 6. Dirac dynamics inside Zurek's open-system setting

## 6.1 The actual replacement

The time-evolution equation $i\hbar\partial_t\Psi=H\Psi$ is a general Hamiltonian form. Choosing a Dirac Hamiltonian instead of the nonrelativistic kinetic Hamiltonian implements the author's intended replacement; keeping this time-evolution notation does not revert to a nonrelativistic model.

For a one-dimensional illustration,

$$
H_D=-i\hbar c\,\alpha\partial_q+\beta mc^2+V(q,W),
$$

where $\alpha^2=\beta^2=I$, $\alpha\beta+\beta\alpha=0$, and $V$ is a Hermitian interaction operator. In multiple dimensions, the corresponding Clifford relations apply to the Dirac matrices. The coordinate $q$ is a declared computational landscape coordinate; the paper does not supply a unique mapping from semantic relationships to a physical spacetime coordinate.

Published work explicitly develops open-system Dirac dynamics in density-matrix and phase-space formulations [R1]. Zurek's environment-induced selection depends on the system–environment interaction and the stability of monitored states [R2]. Consequently, using a Dirac generator in that setting is mathematically meaningful. It does not by itself fix the interaction potential or make all computations independent.

## 6.2 Exact joint and reduced equations

Include the relevant environment and boundary:

$$
H_{\mathrm{tot}}=H_D+H_W+H_B+H_E+H_{VW}+H_{VB}+H_{BE}.
$$

Tensor identity factors are suppressed. The exact joint density evolves by

$$
R_{\mathrm{tot}}(t)=U(t)R_{\mathrm{tot}}(0)U(t)^\dagger,
\qquad i\hbar\dot U=H_{\mathrm{tot}}U.
$$

The retained density is

$$
R(t)=\operatorname{Tr}_E R_{\mathrm{tot}}(t).
$$

This is positive and normalized. It is not automatically a closed differential equation in $R$ alone: a future interaction with a correlated environmental degree of freedom may require its current joint state.

For uncorrelated environmental units that interact once and do not return, one obtains a current-state channel

$$
\mathcal E(R)=\operatorname{Tr}_E\!
\left[U(R\otimes\sigma_E)U^\dagger\right].
$$

Writing $\sigma_E=\sum_r q_r\left|r\right\rangle\left\langle r\right|$ gives Kraus operators $K_{sr}=\sqrt{q_r}\left\langle s\right|U\left|r\right\rangle$. They satisfy $\sum_{sr}K_{sr}^\dagger K_{sr}=I$, proving complete positivity and trace preservation. The repeated-interaction construction and its Markovian assumptions are documented in [R4]. It is a realization of boundary interaction, not a replacement for Dirac dynamics; $U$ is generated by the chosen Dirac-based joint Hamiltonian.

## 6.3 Time-local open-system limit

Under assumptions that justify a time-local Markovian reduced model, use

$$
\dot R=-\frac{i}{\hbar}[H_{DW B},R]
+\sum_\nu\left(L_\nu RL_\nu^\dagger
-\frac12\{L_\nu^\dagger L_\nu,R\}\right).
$$

Nonnegative rates are absorbed into $L_\nu$. This generator has the standard completely positive Markovian form [R5]. The dissipators represent the selected environment and coupling; they are not determined by the word “Zurek” alone. A relativistic system Hamiltonian together with a laboratory-frame environment need not produce a fully Lorentz-covariant dissipator. Full spacetime covariance is an additional physical property, not needed to run the corresponding numerical model.

## 6.4 Explicit einselection and boundary conditioning

Let a boundary observable $A_B$ couple through $A_B\otimes E$. In a dephasing realization,

$$
\mathcal D(R)=-\frac\gamma2[A_B,[A_B,R]],\qquad\gamma\ge0.
$$

In its eigenbasis, $A_B\left|a\right\rangle=a\left|a\right\rangle$, the dephasing part alone gives the following evolution. The same formula holds in the interaction picture when the Hamiltonian commutes with $A_B$:

$$
R_{aa'}(t)=
\exp\!\left[-\frac\gamma2(a-a')^2t\right]R_{aa'}(0).
$$

The diagonals remain unchanged; different pointer values lose coherence. For two eigenvalues separated by $2$, choosing $\gamma t=(\log2)/2$ halves the corresponding coherence.

A measured output $\tau$ has an outcome map $\mathcal J_\tau$ and probability $p_\tau=\operatorname{Tr}\mathcal J_\tau(R)$. Its conditioned state is $\mathcal J_\tau(R)/p_\tau$. Einselection explains stable readout alternatives; the specified instrument determines their probabilities and conditioned states. No individual random outcome is made predictable merely by knowing the density dynamics exactly.

# 7. The boundary connects Feynman amplitudes to environmental selection

## 7.1 One propagator, not an unrelated answer-scoring system

Choose a complete basis of the current joint configurations, including mutable weights and boundary states. Insert the identity between short-time propagators:

$$
\left\langle f\right|U_N\cdots U_1\left|i\right\rangle
=\sum_{s_1,\ldots,s_{N-1}}
\prod_{k=0}^{N-1}\left\langle s_{k+1}\right|U_{k+1}\left|s_k\right\rangle.
$$

This equality is exact for finite matrices. Every represented intermediate configuration contributes coherently. A trajectory is a path of the whole relevant organization, not a path through fixed vectors while its changing rules are omitted.

In an appropriate continuum limit this becomes a Feynman representation. For a Dirac problem, spinor transport and operator ordering must remain in the amplitude; a scalar phase formula alone is generally insufficient. The finite identity avoids ambiguity about the path-integral measure.

## 7.2 Environmental influence is a two-path object

For system paths $\gamma,\gamma'$ that drive environment propagators $U_E[\gamma]$ and $U_E[\gamma']$, tracing out an initially independent environment introduces

$$
\mathcal F[\gamma,\gamma']
=\operatorname{Tr}_E\!
\left(U_E[\gamma]\sigma_EU_E[\gamma']^\dagger\right).
$$

The corresponding density-path contribution is weighted by

$$
e^{\,i(\mathcal A[\gamma]-\mathcal A[\gamma'])/\hbar}
\,\mathcal F[\gamma,\gamma'].
$$

For a pure environmental state, the factor is the overlap of the two resulting environmental states. This is the elementary form of the influence-functional connection [R6]. Boundary-supported interactions determine which path differences the environment can distinguish.

The two paths represent the ket and bra of a density operator. They are not a stored sequence of earlier training activations and do not require reverse-time learning.

## 7.3 Preserving the haystack interference while making an output classical

This is an important constructive condition. An environment that records which candidate path was taken *before those paths recombine* can remove the interference needed for the proposed selection.

For two equal paths entering a dark output port with opposite phases, an environmental overlap $d$ gives

$$
p_{\mathrm{dark}}=\frac{1-\operatorname{Re}d}{2}.
$$

| Environmental overlap | Dark-output probability |
|---:|---:|
| $1$ | $0$ |
| $1/2$ | $1/4$ |
| $0$ | $1/2$ |

The repair is available within the source's own inside/boundary distinction: let the environment monitor **outcome sectors**, not every unresolved internal path.

Let $P_r$ project onto macro-outcome sectors. Set

$$
\Pi_B(R)=\sum_rP_rRP_r,\qquad
\mathcal D_B=\gamma(\Pi_B-I).
$$

Then

$$
e^{t\mathcal D_B}R=
\Pi_B(R)+e^{-\gamma t}\big(R-\Pi_B(R)\big).
$$

**Proof.** $\Pi_B^2=\Pi_B$, so decompose $R$ into its projected component and its complementary component. The first has generator eigenvalue $0$, the second $-\gamma$.

All coherences *within* an outcome sector survive this dissipator, while coherences between distinct sectors decay. With compatible Hamiltonian evolution during readout, this realizes a coherent interior and a stable classical boundary without demanding that decoherence somehow recognize semantically incorrect answers.

# 8. Feynman–Newton inference at two scales

## 8.1 Newtonian dynamics derived from a Dirac band

For a smooth scalar potential and a separated positive-energy band, insert the leading-order ansatz $\Psi=a\,e^{iS/\hbar}$ into the Dirac equation. Neglecting amplitude derivatives at leading semiclassical order gives the dispersion relation

$$
(-\partial_tS-V)^2=c^2\|\nabla S\|^2+m^2c^4.
$$

Choosing the positive-energy branch,

$$
-\partial_tS=H_{\mathrm{pos}}(q,\nabla S),\qquad
H_{\mathrm{pos}}=V(q)+\sqrt{m^2c^4+c^2\|p\|^2}.
$$

Its characteristics obey

$$
\dot q=\frac{c^2p}{\sqrt{m^2c^4+c^2\|p\|^2}},
\qquad \dot p=-\nabla V(q).
$$

For $\|p\|\ll mc$,

$$
H_{\mathrm{pos}}=mc^2+\frac{\|p\|^2}{2m}+V+O\!\left(\frac{\|p\|^4}{m^3c^2}\right),
$$

and therefore $m\ddot q=-\nabla V$ to the corresponding order. Spin transport, band crossings, and strong quantum corrections cannot be discarded outside these assumptions.

The important architectural consequence is that Newtonian motion and the Feynman description can arise from the **same generator and action**. No separate task-cost function has been substituted for the original complex amplitudes.

## 8.2 What stationary phase actually proves

Suppose a scalar channel amplitude has the form

$$
I_k(\varepsilon)=\int a_k(x)e^{iS_k(x)/\varepsilon}\,dx.
$$

Assume smooth amplitudes with compact support and smooth phases. If $\nabla S_k$ never vanishes on the support, repeated integration by parts gives $I_k(\varepsilon)=O(\varepsilon^M)$ for every fixed $M$, with constants depending on the amplitude, phase and derivative bounds.

If one channel has a single nondegenerate stationary point $x_{\mathrm{s}}$ and $a(x_{\mathrm{s}})\ne0$, then

$$
I_{\mathrm{stat}}(\varepsilon)
=(2\pi\varepsilon)^{d/2}
\frac{a(x_{\mathrm{s}})}{\sqrt{|\det S''(x_{\mathrm{s}})|}}
\,e^{iS(x_{\mathrm{s}})/\varepsilon+i\pi\operatorname{sgn}(S'')/4}
+O(\varepsilon^{d/2+1}).
$$

**Needle-separation consequence.** If the desired output has that surviving stationary contribution and all competing channels are nonstationary under the stated conditions, their relative intensity tends to zero. For a fixed finite set of competing channels, normalized output probability tends to one for the surviving channel.

The nonstationary bound follows by applying the differential operator built from $\nabla S/\|\nabla S\|^2$ to the exponential and transferring derivatives to the smooth amplitude; boundary terms vanish by compact support. The stationary term follows by a local quadratic expansion and the oscillatory Gaussian integral. This is a genuine sufficient condition for the proposed needle mechanism—not a guarantee that every input automatically produces this favorable phase geometry.

Multiple stationary contributions can survive or cancel. Stationarity is not synonymous with minimum action or semantic truth. The unresolved learning question is how the density–weight organization comes to give correct answers the required phase structure.

## 8.3 Exact 25-configuration interference calculation

**Chosen realization.** This normalized unitary example checks the all-possibilities/interference mechanism without adding a cost optimizer or postselection.

On $25$ configurations, let

$$
F_{jk}=\frac1{5}e^{2\pi ijk/25},\qquad
(D_r)_{jj}=e^{2\pi irj/25}.
$$

Start from $\left|0\right\rangle$, apply $F$, the bit-specified phase ramp $D_r$, and $F^\dagger$. The output amplitude at configuration $k$ is

$$
a_k=\frac1{25}\sum_{j=0}^{24}e^{2\pi i(r-k)j/25}.
$$

If $k=r$, all terms have the same phase and $a_r=1$. Otherwise the geometric series has ratio $z\ne1$ with $z^{25}=1$, giving

$$
a_k=\frac{1-z^{25}}{25(1-z)}=0.
$$

Thus every possible output is calculated:

| Input $r=7$; candidate output indices | Amplitudes | Probabilities |
|---|---|---|
| 0 through 6 (each) | all $0$ | all $0$ |
| $7$ | $1$ | $1$ |
| 8 through 16 (each) | all $0$ | all $0$ |
| 17 through 24 (each) | all $0$ | all $0$ |

Write $k=5a+b$. The macro group is $a=1$ and the fine configuration is $b=2$. Coarse and fine nested readout therefore identify the same configuration $7$. The calculation was also checked for every $r\in\{0,\ldots,24\}$.

This example proves exact phase-directed selection in a specified unitary system. Here the input explicitly supplies $r$; the machine has not inferred an unknown theorem or learned a textbook. It is a transparent verification of a mechanism, not an intelligence benchmark. The stationary-phase index condition identifies the surviving phase ramp, while the Newtonian trajectory derivation is the distinct continuum result in Section 8.1.

## 8.4 Macro-to-micro trajectory consistency

For a Markovian closed propagator, the composition law is

$$
K_T(q_f,q_i)=\int K_{T-t}(q_f,z)K_t(z,q_i)\,dz.
$$

At stationary phase,

$$
\partial_z\big(S_t(z,q_i)+S_{T-t}(q_f,z)\big)=0.
$$

The derivative is the momentum leaving the first segment minus the momentum entering the second. Its vanishing threads the macro trajectory through a consistent local step. It does not require enumerating independent full paths if the propagator has a reusable representation.

For a free particle with $m=1$, $q_i=0$, $q_f=2$, $T=1$ and $t=1/4$,

$$
S_t(z,0)=\frac{z^2}{2t},\qquad
S_{T-t}(2,z)=\frac{(2-z)^2}{2(1-t)}.
$$

Stationarity gives $z/t=(2-z)/(1-t)$, hence $z=1/2$. The first action is $1/2$, the remaining action is $3/2$, and their sum is $2$, equal to the unsplit trajectory action. The intermediate Hessian is $1/t+1/(1-t)=16/3$.

The endpoint $2$ is specified in this boundary-value example. To predict an endpoint from an initial state instead, propagate an initial wavepacket. A free Gaussian with center $0$, mean momentum $2$, $m=1$, initial position standard deviation $0.1$ and $\hbar=0.01$ has center $2t$ and variance $0.01+0.0025t^2$. At $t=1$, it is centered at $2$ with standard deviation $\sqrt{0.0125}$. A sharp coarse trajectory is a concentration property of that state, not a universal assumption about all inputs.

For open dynamics with environmental memory, the reduced propagator need not compose this way. Composition is exact on the retained state including the necessary boundary modes. This is where the original present-state principle matters to the inference calculation itself.

## 8.5 Coarse and fine output measurements

If $\{P_j\}$ is a macro partition and $\{Q_{a|j}\}_a$ are orthogonal fine projectors with $\sum_aQ_{a|j}=P_j$, then

$$
p(j)\,p(a\mid j)=\operatorname{Tr}(Q_{a|j}R).
$$

This follows from $P_jQ_{a|j}=Q_{a|j}$ and cyclicity of trace. It establishes exact two-stage readout for nested partitions of the same state. If additional dynamics occurs between readouts, its measurement back-action and changed generator must be included; the equality cannot be reused as though nothing changed.

# 9. An exact vector–weight–output–feedback calculation

This example connects several mechanisms in a single calculation. It is a finite two-component, two-mode realization of a massless Dirac kinetic term with a Hermitian mode coupling, not a complete multimodal intelligence or an exact untruncated continuum field. All equations below are exact for this declared finite Hamiltonian. The coefficients and retained modes are explicitly chosen.

## 9.1 Joint evolution

Let $Q$ be the information spinor and $W$ a two-state relational/momentum control. In units with $\hbar=1$,

$$
\mathbb H=X_Q\otimes Z_W+I_Q\otimes X_W.
$$

The first term is a Dirac kinetic-axis coupling with its signed coefficient represented by a weight observable. The second term changes that weight state. The two terms anticommute and square to $I$, so $\mathbb H^2=2I$ and

$$
U(t)=\cos(\sqrt2t)I-
\frac{i\sin(\sqrt2t)}{\sqrt2}\mathbb H.
$$

Starting at $\left|00\right\rangle$, choose $t_{\mathrm{s}}=\pi/(4\sqrt2)$. Then

$$
\left|\Psi\right\rangle=\frac{\left|00\right\rangle}{\sqrt2}
-\frac{i\left|01\right\rangle}{2}-\frac{i\left|10\right\rangle}{2}.
$$

Both reduced densities equal

$$
R_Q=R_W=\begin{pmatrix}
3/4&i/(2\sqrt2)\\-i/(2\sqrt2)&1/4
\end{pmatrix}.
$$

Their purity is

$$
\operatorname{Tr}(R_Q^2)=\frac9{16}+\frac1{16}+\frac14=\frac78.
$$

The full joint state is pure, but its marginals are not. This illustrates why incomplete local knowledge and relational correlation belong in the same joint density description.

## 9.2 The correlations change the actual differential equation

The exact identity is

$$
\frac{d}{dt}\left\langle Z_Q\right\rangle=2\left\langle Y_QZ_W\right\rangle.
$$

At $t_{\mathrm{s}}$,

$$
2\left\langle Y_QZ_W\right\rangle=-\sqrt2.
$$

Replacing the joint density by the product of its marginals instead gives

$$
2\left\langle Y_Q\right\rangle\left\langle Z_W\right\rangle
=2\left(-\frac1{\sqrt2}\right)\left(\frac12\right)
=-\frac1{\sqrt2}.
$$

The product approximation misses half the instantaneous response. This quantitatively supports the author's insistence that vectors and weights must be represented relationally rather than as unrelated scalar multipliers.

## 9.3 Boundary output and immediate re-entry

Let the environment select the $Z_Q$ readout sectors. Here readout is an ideal boundary event after $U(t_{\mathrm{s}})$, not a claim that continuous dephasing under the same noncommuting interaction produces an exact projective measurement in finite time. A complete readout emits bit $\tau=0$ or $1$. Conditioning gives

$$
p_0=\frac34,\qquad
\left|w_0\right\rangle=\sqrt{\frac23}\left|0\right\rangle-i\sqrt{\frac13}\left|1\right\rangle,
$$

$$
p_1=\frac14,\qquad \left|w_1\right\rangle=\left|0\right\rangle.
$$

The joint post-readout states are $\left|0\right\rangle\left|w_0\right\rangle$ and $\left|1\right\rangle\left|w_1\right\rangle$. The weight expectations before feedback are $\left\langle Z_W\right\rangle_0=1/3$ and $\left\langle Z_W\right\rangle_1=1$.

**Chosen input coupling.** Every re-entered bit $b$ drives the same local rotation on both vector and weight:

$$
V_b=\exp\!\left[-i(2b-1)\frac\pi4X\right],\qquad
\mathcal E_b(R)=(V_b\otimes V_b)R(V_b\otimes V_b)^\dagger.
$$

These are two bit-controlled Hamiltonian pulses of opposite sign. No reverse pass is used. Both output branches now have the *same vector state*, with Bloch vector $(0,1,0)$, but different weight states:

| Re-entered output | Output probability | Weight Bloch vector after feedback |
|---|---:|---|
| $0$ | $3/4$ | $(0,1/3,2\sqrt2/3)$ |
| $1$ | $1/4$ | $(0,-1,0)$ |

Consequently, immediately after feedback,

$$
\left.\frac{d\left\langle Z_Q\right\rangle}{dt}\right|_{\tau=0}
=\frac{4\sqrt2}{3}\approx1.885618083,
\qquad
\left.\frac{d\left\langle Z_Q\right\rangle}{dt}\right|_{\tau=1}=0.
$$

The vector's differential response differs because its relational density differs, even though its own marginal state is identical in the two branches.

## 9.4 The next emitted bit also changes

Run the same $U(t_{\mathrm{s}})$ again and read the same $Z_Q$ observable. Direct multiplication gives

$$
U^\dagger Z_QU
=\frac12 Z_Q+\frac12Y_QY_W+\frac1{\sqrt2}Y_QZ_W.
$$

Since the post-feedback vector has $\left\langle Y_Q\right\rangle=1$ and $\left\langle Z_Q\right\rangle=0$, the two next-step expectations are

$$
\left\langle Z_Q\right\rangle_{\mathrm{next}\mid0}
=\frac12\frac13+\frac1{\sqrt2}\frac{2\sqrt2}{3}=\frac56,
$$

$$
\left\langle Z_Q\right\rangle_{\mathrm{next}\mid1}=-\frac12.
$$

Therefore

$$
P(\tau_2=0\mid\tau_1=0)=\frac{11}{12},\qquad
P(\tau_2=0\mid\tau_1=1)=\frac14.
$$

All two-output probabilities are:

| Output pair | Exact probability |
|---|---:|
| $00$ | $11/16$ |
| $01$ | $1/16$ |
| $10$ | $1/16$ |
| $11$ | $3/16$ |

They sum to one. Five successive output/feedback steps, including all $32$ possible output branches, were also checked for density positivity and total probability one.

**What this proves.** An output can immediately reshape the joint density, alter the differential dynamics of another represented degree of freedom, and change subsequent output probabilities without a saved output history. It does not prove that the chosen pulses learn meaningful language; those pulses are the declared numerical realization.

# 10. The complete present-state event loop

The following theorem connects the input, inference, boundary readout, re-entry and compilation parts of the source.

## 10.1 Event maps

Let $\mathcal B_b$ describe the interaction/observation associated with the next external bit $b$. It can be trace-preserving for a controlled injection, or trace-nonincreasing for conditioning on an observed bit in a predicted boundary stream. Define

$$
q_b=\operatorname{Tr}\mathcal B_b(R),\qquad
R_b=\mathcal B_b(R)/q_b,
$$

whenever $q_b>0$. All examples explicitly state which interpretation applies.

Let $\{\mathcal J_{\tau}\}$ be the output instrument. Its maps include the current coherent evolution, the actual coarse/fine interactions, and the boundary readout. Their sum is trace-preserving. Feynman path sums are a representation of the coherent evolution inside this instrument; they are not an additional stochastic answer generator.

Let $\mathcal E_\tau$ be the same machine's bit-coupling channel when its emitted bit re-enters. Let $\Pi$ be a trace-preserving, completely positive canalization map.

Then

$$
p(\tau\mid R,b)=\operatorname{Tr}\mathcal J_\tau(R_b),
$$

$$
R^{\mathrm{new}}_{b,\tau}=
\Pi\!\left[
\mathcal E_\tau\!\left(
\frac{\mathcal J_\tau(R_b)}{p(\tau\mid R,b)}
\right)\right].
$$

Combining the stages,

$$
R^{\mathrm{new}}_{b,\tau}
=\frac{\Pi\mathcal E_\tau\mathcal J_\tau\mathcal B_b(R)}
{\operatorname{Tr}(\mathcal J_\tau\mathcal B_b(R))}.
$$

Here juxtaposition means composition, read from right to left. The maps may depend on classical structural variables already included in the complete current state. If control degrees of freedom are represented quantum mechanically, they should normally be included in $R$ itself; replacing them by expectations is not silently assumed. If an interaction is instead selected nonlinearly from the current density, positivity can still hold pointwise, but the overall map is not automatically a linear quantum channel. Channel contraction claims below apply only when the compared states undergo the same declared channel, or when the control is included in a common joint channel.

## 10.2 Event-loop theorem

**Assumptions.** All maps are defined on the represented state spaces; $\mathcal B_b$ and each $\mathcal J_\tau$ are completely positive; $\sum_\tau\mathcal J_\tau$, $\mathcal E_\tau$, and $\Pi$ preserve trace; the conditioning events considered have nonzero probability; and no omitted environment later returns with unrepresented relevant correlations.

**Conclusions.** Every admitted event produces a positive, normalized state. Output probabilities sum to one. Repeated events define a controlled Markov process on the complete present configuration. Output feedback changes this process through the same current state; no transcript is needed to evaluate the next event.

**Proof.** A composition of completely positive maps is completely positive, so the numerator is positive. Trace preservation of $\Pi$ and $\mathcal E_\tau$ makes its trace equal to the denominator. Normalization follows. Since the output instrument sums to a trace-preserving map,

$$
\sum_\tau p(\tau\mid R,b)=\operatorname{Tr}R_b=1.
$$

At the next event, all maps and probabilities are determined by the new complete state and the next supplied bit. Two earlier event histories yielding that same state therefore yield the same next-event distribution. This is the controlled Markov property. Induction gives the same statement for any finite future input sequence. The argument does not require a fixed upper bound on the number of events.

A zero-probability observation cannot be normalized in a model that excluded it. The universal physical bit input can still be injected, but its *interpretation* must remain unresolved or the represented space must be revised. The density formalism does not justify inventing a posterior by dividing by zero. The original distinction between unknown possibilities and hard constraints is essential here.

## 10.3 Why feedback is not automatically new external evidence

The source's sensory and active blanket ports already distinguish incoming environmental information from an emitted action. The same bit value can act through the same elementary interaction algebra at either port while having a different causal origin.

An emitted bit physically changes the present state. That is sufficient for metamorphosis. It does not, solely by being re-entered, become an independent observation confirming the machine's own belief. This is a consequence of conditioning on the current state, not a proposed extra supervisor. The event equations above separate the output instrument from the feedback interaction so the same observed output is not counted twice as independent evidence.

# 11. Canalization, idempotent locks, and retention

## 11.1 The lock can arise from the dynamics

The paper explicitly proposes many-to-one canalization and $\Pi^2=\Pi$. In finite dimension, suppose a dissipative semigroup $e^{t\mathcal L}$ converges as $t\to\infty$. Define

$$
\Pi=\lim_{t\to\infty}e^{t\mathcal L}.
$$

Then $\Pi$ is completely positive and trace-preserving, and

$$
\Pi^2=\lim_{t,s\to\infty}e^{(t+s)\mathcal L}=\Pi.
$$

A sufficient spectral condition is that zero is the only eigenvalue on the imaginary axis and all other eigenvalues have negative real part. Possible nonzero imaginary modes must be retained as ongoing dynamics rather than falsely described as a settled lock. The boundary dephasing channel in Section 7 gives an explicit example with exponential convergence.

## 11.2 A relation-preserving many-to-one map

Suppose a current region factors as a relational subsystem $R$ and a nuisance subsystem $N$. For a fixed normalized nuisance state $\sigma_N$, define

$$
\Pi(\rho)=\operatorname{Tr}_N\rho\otimes\sigma_N.
$$

It is trace-preserving and completely positive, and applying it twice changes nothing. For any relational observable $A_R$,

$$
\operatorname{Tr}\!\left[\Pi(\rho)(A_R\otimes I_N)\right]
=\operatorname{Tr}\!\left[\rho(A_R\otimes I_N)\right].
$$

Thus a whole class of different fine configurations becomes one current realization with exactly the same relational predictions. This is genuine compaction of irrelevant distinctions, not a disguised transcript database.

## 11.3 Retention of future behavior: the required algebra

Idempotence alone does not guarantee that discarded information will never matter again. The precise condition is most useful on the *unnormalized outcome maps*. For every relevant input/output event map $\mathcal K$ require

$$
\Pi\mathcal K=\Pi\mathcal K\Pi.
$$

Then, for any finite sequence of these maps,

$$
\Pi\mathcal K_n\cdots\mathcal K_1(\rho)
=\Pi\mathcal K_n\Pi\cdots\Pi\mathcal K_1\Pi(\rho).
$$

**Proof.** Repeatedly replace $\Pi\mathcal K$ by $\Pi\mathcal K\Pi$. Since $\Pi$ preserves trace, both sides have identical sequence probabilities. For nonzero probabilities, their normalized retained states also coincide.

This proves that compilation after each event preserves all future outcome probabilities covered by the condition. It is the exact mathematical requirement underlying “keep the functionality, not the history.” No separate checker is part of the architecture; the equation is a property the proposed dynamics must satisfy.

For continuous evolution, $\Pi\mathcal L=\Pi\mathcal L\Pi$ yields a closed equation for the compiled state and implies the corresponding semigroup identity by uniqueness of the finite-dimensional linear differential equation.

## 11.4 Preserving old structure while adding new distinctions

Repeated locks can be incompatible. For example, dephasing a qubit first in the $Z$ basis and then in the $X$ basis can erase previously retained $Z$ populations. Calling both maps idempotent does not prevent that loss.

The source's hierarchy supplies a constructive alternative: represent a retained relation in a nested subspace instead of overwriting its distinguishing degree of freedom. An isometry $V$ with $V^\dagger V=I$ gives

$$
\rho^{\mathrm{new}}=V\rho V^\dagger,\qquad A^{\mathrm{new}}=VAV^\dagger,
$$

and

$$
\operatorname{Tr}(\rho^{\mathrm{new}}A^{\mathrm{new}})=\operatorname{Tr}(\rho A).
$$

It can add representational space without storing the earlier transcript. Later dynamics must still preserve the relevant relational behavior; the isometry alone does not prove indefinite skill retention. Growth also has a storage cost. “No history tape” and “unlimited independently recoverable knowledge in a fixed number of bits” are different claims.

## 11.5 Current-state storage versus chronological storage

For a fixed represented dimension $D$ and fixed numerical precision, the event loop can reuse the same state storage for arbitrarily many updates. This establishes memory overhead independent of the number of processed events. If the organization develops more independent distinctions, its dimension, precision, or compressed representation can change.

A classical implementation with $B$ total retained bits has at most $2^B$ configurations. Exact recovery of $n$ arbitrary independently supplied bits requires at least $2^n$ distinguishable configurations, hence $B\ge n$. This bound concerns independent recoverability, not the ability of a compact law to generate indefinitely many structured consequences. The proposal's structural compaction is meaningful precisely where many histories share the same useful underlying organization.

# 12. Forward structural learning and the textbook hypothesis

The paper proposes that structured experience progressively changes the computational landscape. Two conditional results make that statement calculable within its own mechanisms.

## 12.1 A dissipative correlation-to-weight law from a shared interaction

Let $w$ be a slowly varying relational coordinate and $C$ a bounded vector-correlation observable. Choose interaction energy

$$
E_{\mathrm{int}}(w,C)=\frac\kappa2w^2-gwC,\qquad\kappa>0.
$$

The vectors experience the $-gwC$ term in their generator, while the force on $w$ is $gC-\kappa w$. This is reciprocity derived from one coupling, not two unrelated update rules.

In an overdamped, Markovian bath limit with friction $\gamma>0$, the corresponding classical weight equation is

$$
dw=\frac{g\left\langle C\right\rangle-\kappa w}{\gamma}\,dt
+\sqrt{\frac{2k_BT}{\gamma}}\,dB_t.
$$

This is a mean-field, slow-coordinate approximation to the coupled problem, not an exact replacement for the joint density equation. It makes the paper's forward plasticity intuition explicit. An assumed thermal bath supplies both damping and fluctuations; discarding fluctuations requires a zero-noise idealization or a stated approximation.

For constant correlation $c_0$, deterministic $w(0)=w_0$, and the stated noise model,

$$
\mathbb E[w(t)]
=\frac{gc_0}{\kappa}
+\left(w_0-\frac{gc_0}{\kappa}\right)e^{-\kappa t/\gamma},
$$

$$
\operatorname{Var}(w(t))
=\frac{k_BT}{\kappa}\left(1-e^{-2\kappa t/\gamma}\right).
$$

For $g=\kappa=\gamma=1$, $c_0=0.8$, $w_0=0$, and $k_BT=0.02$:

| Time | Mean relation strength | Variance |
|---|---:|---:|
| $0$ | $0$ | $0$ |
| $\log2$ | $0.4$ | $0.015$ |
| $t\to\infty$ | $0.8$ | $0.02$ |

A recurrent relation thus changes a retained dynamical parameter through forward interaction. No replay of the observations is needed to evaluate the next differential step. Whether $C$ captures a meaningful mathematical or linguistic relation is a property to be demonstrated by the system's input-generated organization.

## 12.2 Structured recurrence stabilizes a representation under a gap condition

Suppose one recurring lesson's resulting unconditioned channel has the form

$$
\mathcal F=r\,\mathrm{Id}+(1-r)\Pi_L,\qquad 0\le r<1,
$$

where $\Pi_L$ is a completely positive, trace-preserving idempotent channel associated with that lesson's stable structure. This is a possible channel produced by the specified interactions, not an assumed universal learning algorithm.

Because $\Pi_L^2=\Pi_L$,

$$
\mathcal F^n(\rho)=\Pi_L(\rho)+r^n\big(\rho-\Pi_L(\rho)\big).
$$

For $r=1/2$, the transient multiplier after $1,2,5,10$ repetitions is respectively $1/2,1/4,1/32,1/1024$. This proves consolidation of the channel's stable distinctions. It does not imply that every repeated sentence is true or that the limiting structure is semantically useful. Repetition can consolidate a mistaken or uninformative organization too.

## 12.3 A conditional generalization guarantee

Let $R$ and $R'$ be two current representations and define trace distance

$$
D(R,R')=\frac12\|R-R'\|_1.
$$

For the same subsequent trace-preserving physical process and any one final output event,

$$
|p(\tau\mid R)-p(\tau\mid R')|\le D(R,R').
$$

**Proof.** Trace distance contracts under completely positive, trace-preserving maps. For a final measurement effect $0\le M\le I$, the variational characterization of trace distance bounds $|\operatorname{Tr}(M\Delta R)|$ by $D$. Adaptive output feedback is included by regarding the output instrument and feedback as one channel with a classical output factor.

This says what a learned structural equivalence would buy: unfamiliar inputs that lead to nearby relevant organizations produce nearby predictions under the same subsequent process. It does not prove that the machine has learned to place equivalent meanings nearby. Conditioning on an unlikely observation can also amplify differences; the unconditional contraction bound cannot be applied unchanged after arbitrary postselection.

**The textbook hypothesis is therefore testable in three connected parts:** ordered information must create appropriate recurring correlations; those correlations must change and stabilize the density–weight organization; and unfamiliar arrangements expressing the same relations must engage that organization successfully. The first two have explicit mechanisms and conditional calculations here. The third has a consequence theorem once structural equivalence is established, but has not been demonstrated on textbooks in this report.

# 13. Hierarchy, hard constraints, locality, and bit-level execution

## 13.1 Hierarchical densities without pretending independent factors are enough

A product $\bigotimes_k\rho_k$ is valid when the factors are independent. The author's Part 9 explicitly asks for more. A joint density or its tensor-network representation is required when correlations matter, as Section 9's factor-of-two derivative error shows.

A globally positive representation is $R=XX^\dagger/\operatorname{Tr}(XX^\dagger)$ with $X$ stored in a tensor network. Hierarchical isometries can reorganize this factorization. They change the representation without automatically discarding the correlations.

If a coarse isometry $V$ represents only a subspace $P=VV^\dagger$ of a fine space, the expression $V^\dagger RV$ alone has trace $\operatorname{Tr}(PR)$, not necessarily one. A trace-preserving representation can retain a complementary “not represented in this coarse sector” outcome:

$$
\mathcal C(R)=V^\dagger R V
\ \oplus\ \operatorname{Tr}[(I-P)R] \,\left|\perp\right\rangle\left\langle\perp\right|.
$$

This is a completely positive channel into the coarse space plus a one-dimensional remainder flag. It intentionally removes cross-sector coherence; exact future preservation therefore requires the canalization condition, not just renormalizing the surviving part and forgetting the remainder.

A coherent superposition of *mixed density descriptions* requires an explicit enlarged state, such as chosen purifications plus a layer label. The mixture $p\rho_0+(1-p)\rho_1$ is not itself evidence of coherent cross-layer interference. The source's hierarchy allows the larger construction; its phases and correlations must be represented rather than inferred from the two marginal matrices alone.

## 13.2 Making null-space constraints persistent

Let $P$ project onto allowed configurations. If

$$
R=PRP,\qquad H=PHP,\qquad L_\nu=PL_\nu P,
$$

then both the Hamiltonian and dissipative terms remain supported in $P$. Thus no probability enters the forbidden complement. More general invariant-subspace conditions are possible, but this sufficient condition is immediate and checkable.

A zero density eigenvalue alone says a direction has no current population. It does not prohibit a generator from populating that direction later. Representing a hard impossibility through both support and invariant dynamics supplies the persistence intended by the paper. Unobserved but possible configurations should not be classified as hard-negative merely because they currently lack evidence.

## 13.3 Locality has a precise, non-global meaning

For a local Dirac equation with suitable regular coefficients and boundary conditions, the principal first-order differential part controls finite propagation speed. An input changes the global mathematical state immediately because that state includes the affected region; it need not change every distant local observable instantly.

A continuous-time lattice Hamiltonian with nearest-neighbor terms generally has small nonzero tails beyond a nominal lattice cone. A discretization made from local shift and local rotation steps can instead have an exact finite event cone: after $n$ steps, a perturbation reaches at most $n$ shift edges. Its continuum limit may approximate the Dirac equation, but it is not identical to the continuum equation at finite step size.

Independent local operations commute when they act on disjoint tensor factors. Overlapping operations generally do not. Thus asynchronous execution must respect actual dependencies. The author’s local-update objective does not require an all-to-all synchronization barrier, but it also does not imply that coupled regions can be advanced in arbitrary order without changing the result.

## 13.4 Work and storage actually required by bit execution

Bits can encode matrix entries, amplitudes, coefficients and the instruction to evaluate a differential update. They do not turn one full matrix update into one Boolean gate.

A dense $D\times D$ complex density needs $D^2$ stored complex entries. A pure-state matrix-product representation with $N$ sites, local dimension $d$ and maximum bond dimension $\chi$ uses order $Nd\chi^2$ entries. A locally purified density representation has an additional local purification dimension. Exact representation requires sufficient ranks across the relevant cuts; maintaining small rank is a property of the evolving state, not a consequence of merely choosing Dirac dynamics.

For a scalar coherent sum $A=\sum_j a_j e^{iS_j/\hbar}$ with fixed amplitudes, phase-action error $|\widetilde S_j-S_j|\le\delta$ yields

$$
|\widetilde A-A|\le\frac\delta\hbar\sum_j|a_j|.
$$

**Proof.** Use $|e^{ix}-e^{iy}|\le|x-y|$ and the triangle inequality. Hence strong cancellation can require high precision: a small final amplitude can have a large relative error even when each local phase error is small.

For $H=H_1+H_2$, a first-order split step has local operator error $O(h^2\|[H_1,H_2]\|/\hbar^2)$ in the bounded-matrix setting. Higher-order splitting or exact local exponentials improve the numerical realization without changing the proposed physical mechanism. No hardware throughput figure is established here without a complete update kernel and measured memory traffic.

# 14. Readable algorithm for the coupled machine

The following is an algorithm in mathematical instructions, not program code. Its primitives are the paper's bit interactions, density evolution, boundary communication, coherent paths, stationary-action approximation, output re-entry, and canalization. Numerical choices needed by a specific run are stated in advance rather than hidden inside the terminology.

**1. Retain the present organization.** Maintain the current joint density, its hierarchical representation, and the structural coordinates needed to define its local Dirac interactions and boundaries. Do not keep an input transcript merely for replay.

**2. Accept the next available bit at the sensory boundary.** A bit may be processed before a complete word, image object, formula or sound unit is identified. The binary substrate and elementary operation algebra remain the same across sources.

**3. Evolve or condition the boundary state using that bit.** Apply the specified input interaction. Where an observation conditions a predicted stream, normalize by its calculated probability. Preserve simultaneous complete and incomplete interpretation sectors. An unsupported observation remains unresolved; it is not assigned invented certainty.

**4. Update the relevant joint vector–weight density.** Include the current correlations that influence the active region. Apply the bit-derived operator through a positivity-preserving joint map, not by an arbitrary product of density matrices.

**5. Determine the active interaction boundary.** Identify the region's current interior, boundary and exterior from the actual generator and retained state. Where an exact statistical blanket is asserted, check its conditional-independence condition. When correlations cross the proposed cut, retain the necessary boundary/joint degrees of freedom rather than pretending they are absent.

**6. Evolve by the Dirac-based open dynamics.** Use the current joint Hamiltonian, including relational-control operators and boundary environmental couplings. Derive vector and weight changes from that same interaction. Advance only causally affected regions, respecting the order of dependent events.

**7. Form the macro inference amplitude.** Starting from the current density, represent the coherent propagation through the available joint configurations. Every path includes the weight and boundary changes that would affect it. Use exact operator propagation or a controlled path representation; do not convert the complex phases into unrelated positive path costs.

**8. Identify the stationary macro contributions.** When a semiclassical approximation is justified, derive the characteristic trajectories from that same action. Retain multiple surviving contributions when no unique dominant one has been established. A stationary solution is not labeled correct merely because it is stationary.

**9. Refine to the nearest output-producing transition.** Resolve the local frontier and compose its action with the macro trajectory. Match intermediate momenta or use the exact nested amplitude/instrument. Account for the back-action of any macro measurement actually performed.

**10. Allow boundary selection without destroying needed interior coherence.** Environmental coupling should distinguish the chosen output sectors while preserving the unresolved path interference needed inside a sector. Use the density calculation to determine the resulting readout probabilities.

**11. Emit the first atomic output configuration.** Apply the declared boundary measurement instrument. The observed bit or configuration is the output event; update the density by conditioning on that event once.

**12. Re-enter that output immediately.** Feed its bit configuration through the machine's input interaction algebra at the active/feedback port. Modify vectors, relational controls and any affected boundary structure before moving to the next external bit. Do not wait until the whole answer has been emitted.

**13. Canalize the resulting organization.** Let stable structure settle or apply its justified compiled projection. Remove only distinctions whose removal preserves the relevant future process, as expressed by the canalization identity. Preserve unresolved or dynamically returning influences.

**14. Reuse or extend the hierarchy.** Reorganize the current density factors. Use isometric nesting for additional distinctions and declared coarse-graining for redundant ones. Include the actual representation and numerical precision in storage accounting.

**15. Continue in the author's order.** After the output has re-entered and the state has changed, take the next original external bit and repeat. A closed/internal settling step can use an identity external injection only when such a step is part of the declared event protocol. When the external stream ends, any continued answer emission or physical settling requires a specified boundary stopping condition; it is not inferred from the absence of bytes alone.

The sequence describes phases of one coupled interaction process. It does not posit fifteen independent cognitive faculties or an external planner controlling the physical equations.

# 15. What has been proved, what has been calculated, and what remains

## 15.1 Proved results

The report provides direct proofs for positivity and normalization of the incomplete-input instrument; the generator-level boundary influence equation; the Gaussian spiral's stationary Markov blanket; reciprocal vector–weight commutator equations; the finite event-loop Markov property; boundary sector dephasing and idempotence; the exact 25-configuration interference identity; macro–micro action matching; relation-preserving canalization and future-event equivalence; isometric retention; support-invariant hard constraints; and the specified consolidation and generalization bounds. The leading semiclassical calculation and stationary-phase theorem explicitly state their approximation assumptions.

The quantum conditional-independence characterization is an established external theorem [R3], not a newly proved result of this report. The Dirac open-system and completely positive generator frameworks are external mathematical foundations [R1, R4, R5]. Their use here is distinguished from the new chosen examples and source-constrained synthesis.

## 15.2 Symbolic and numerical checks against the written formulas

A deterministic verification run performed **434 symbolic and numerical checks**. These are arithmetic and consistency checks, not 434 independent intelligence experiments. They included all 25 phase-ramp inputs; the exact two-mode joint evolution; commutator identities; ambiguous-prefix probabilities; output feedback and next-output probabilities; the four-dimensional stationary covariance calculation; randomized positivity and projection checks; and all 32 branches of five successive feedback outputs.

Representative exact results are collected here:

| Calculation | Result |
|---|---|
| Completion probability after $0$, $01$, $010$ in the declared example | $1/2$, $3/4$, $2/3$ |
| Marginal purity after the joint Dirac-sector interaction | $7/8$ |
| Exact versus independent-marginal vector derivative | $-\sqrt2$ versus $-1/\sqrt2$ |
| Next zero probability after re-entered zero versus one | $11/12$ versus $1/4$ |
| Two-output probabilities $00,01,10,11$ | $11/16,1/16,1/16,3/16$ |
| Exact 25-configuration interference | One output of probability $1$, the other 24 of probability $0$ |
| Free trajectory's first quarter-time location | $1/2$ |
| Split versus unsplit action | $1/2+3/2=2$ |
| Markov-blanketed spiral eigenvalues | $-2\pm2i$ |
| Spiral radius multiplier at $t=\log2$ | $1/4$ |
| Correlation-driven weight mean and variance at $t=\log2$ | $0.4$ and $0.015$ |

Floating-point verification produced residual wrong-channel amplitudes no larger than approximately $1.3\times10^{-15}$ in the displayed $r=7$ Fourier test. The exact theorem gives zero; numerical residuals are not additional physical possibilities.

## 15.3 Source-check decisions

| Potential reinterpretation | Decision after comparison with the source and author clarifications |
|---|---|
| Replace complex amplitudes by softmax or shortest-path costs | Not done. Keep complex interference and derive the classical limit of the same action. |
| Use Dirac only as a metaphor for local propagation | Not done. Use a Dirac Hamiltonian inside the open-system calculation. |
| Treat a Markov blanket as only an output label | Not done. Derive communication and statistical-boundary conditions explicitly. |
| Require complete words before processing | Not done. Calculate uncertain completion inside the density state. |
| Replace relational densities by independent weights | Not done. Preserve the joint density and quantify the lost response under factorization. |
| Update the machine only after a complete answer | Not done. Re-enter the first output before the next external bit. |
| Add a historical memory store to rescue sufficiency | Not done. Retain only present boundary/relational degrees of freedom that still affect the future. |
| Treat every formal equation as a uniquely determined law | Not done. Mark example couplings, initialization and measurement assignments as choices. |
| Assume a stationary path is automatically a correct answer | Not done. Prove a sufficient phase-separation condition and state the remaining learning claim. |
| Regard the spiral and black-hole comparisons as either identical laws or useless metaphors | Neither. Construct a spiral blanket exactly and isolate the observable-equivalence content of the black-hole analogy. |

## 15.4 The exact outstanding scientific claim

The author has specified a connected set of mechanisms. What is not yet determined uniquely by those statements is the information-dependent interaction law that makes *particular meanings* correspond to stable relations and makes *correct answers* correspond to selected trajectories across unrestricted inputs.

For example, $H_D+gV$ with two different nonzero values of $g$ can satisfy the same locality and open-system structural assumptions but produce different output probabilities. The principles alone do not select the numerical coupling, the initial relational state, or the output observable. Supplying one explicit realization is therefore necessary for a computation; claiming it is the unique realization would be unjustified.

The next substantive proof would establish that a specified bit interaction and developmental coupling causes structured experience to create the required answer-selecting geometry, and that this learned geometry transfers to unfamiliar input arrangements. The current report does not disguise that remaining claim as either an impossibility result or an already demonstrated capability.

**Conclusion.** The proposed loop has a coherent mathematical realization and nontrivial exact examples. Its most important supported features are current-state execution, incomplete-configuration density representation, boundary-mediated influence, reciprocal vector–weight evolution, interference-based selection in specified phase geometries, and immediate output-driven transformation. General intelligence, universal answer correctness and a scalable advantage remain claims to establish through a fully specified developmental law and corresponding tests.

# References and source provenance

**[S1] Author's source.** *Foundational Principles — The Variational Action Machine & Zero-Memory Architecture*, supplied DOCX, 16-page indexed version. The source's master loop and Parts 1–10 are used throughout. Relevant later clarifications are the author's conversation statements about Dirac inside Zurek, bits as the immutable substrate, unknown input completion, the organizing role of Markov blankets, and the external-input/output-re-entry order. Earlier assistant-generated reports are not treated as authority for the author's intent.

**[R1] Dirac open-system dynamics.** R. Cabrera, A. G. Campos, D. I. Bondar and H. A. Rabitz, “Dirac open-quantum-system dynamics: Formulations and simulations,” *Physical Review A* 94, 052111 (2016). DOI: 10.1103/PhysRevA.94.052111. Author manuscript: https://arxiv.org/abs/1409.1247.

**[R2] Decoherence and einselection.** W. H. Zurek, “Decoherence, einselection, and the quantum origins of the classical,” *Reviews of Modern Physics* 75, 715 (2003). DOI: 10.1103/RevModPhys.75.715. Author manuscript: https://arxiv.org/abs/quant-ph/0105127. Used for the physical interpretation of environment-selected pointer states, not as a proof of computational semantics.

**[R3] Exact quantum Markov structure.** P. Hayden, R. Jozsa, D. Petz and A. Winter, “Structure of states which satisfy strong subadditivity of quantum entropy with equality,” *Communications in Mathematical Physics* 246, 359–374 (2004), Theorem 6. DOI: 10.1007/s00220-004-1049-z. https://arxiv.org/abs/quant-ph/0304007.

**[R4] Current-state open-system realizations.** F. Ciccarello, S. Lorenzo, V. Giovannetti and G. M. Palma, “Quantum collision models: Open system dynamics from repeated interactions,” *Physics Reports* 954, 1–70 (2022). DOI: 10.1016/j.physrep.2022.01.001. https://arxiv.org/abs/2106.11974. Relevant sections describe fresh-ancilla Markov assumptions, maps and conditioned trajectories.

**[R5] Completely positive Markovian generators.** G. Lindblad, “On the generators of quantum dynamical semigroups,” *Communications in Mathematical Physics* 48, 119–130 (1976). DOI: 10.1007/BF01608499. https://link.springer.com/article/10.1007/BF01608499.

**[R6] Environmental path influence.** R. P. Feynman and F. L. Vernon, Jr., “The theory of a general quantum system interacting with a linear dissipative system,” *Annals of Physics* 24, 118–173 (1963). DOI: 10.1016/0003-4916(63)90068-X. Caltech record: https://authors.library.caltech.edu/records/x858p-sg795. The elementary overlap formula in this report is derived directly from partial tracing; general influence-functional reductions require their own environment assumptions.

**[R7] Scope of black-hole uniqueness.** P. T. Chruściel, J. Lopes Costa and M. Heusler, “Stationary Black Holes: Uniqueness and Beyond,” *Living Reviews in Relativity* 15, 7 (2012). DOI: 10.12942/lrr-2012-7. https://arxiv.org/abs/1205.6112. Used only to distinguish restricted exterior uniqueness from unrestricted storage or recovery of interior information.
