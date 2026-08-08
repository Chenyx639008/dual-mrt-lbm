# Phase-ﬁeld-based lattice Boltzmann modeling of large-density-ratio two-phase ﬂows

Hong Liang, 1 Jiangrong Xu, 1 Jiangxing Chen, 1 Huili Wang, 2 Zhenhua Chai, 2 , 3 and Baochang Shi 2 , 3 , * 1

Department of Physics, Hangzhou Dianzi University, Hangzhou 310018, China 2 School of Mathematics and Statistics, Huazhong University of Science and Technology, Wuhan 430074, China 3 Hubei Key Laboratory of Engineering Modeling and Scientiﬁc Computing, Huazhong University of Science and Technology, Wuhan 430074, China

![](<liang2018_images/imageFile1.png>)

(Received 3 November 2017; published 19 March 2018)

In this paper, we present a simple and accurate lattice Boltzmann (LB) model for immiscible two-phase ﬂows, which is able to deal with large density contrasts. This model utilizes two LB equations, one of which is used to solve the conservative Allen-Cahn equation, and the other is adopted to solve the incompressible Navier-Stokes equations. A forcing distribution function is elaborately designed in the LB equation for the Navier-Stokes equations, which make it much simpler than the existing LB models. In addition, the proposed model can achieve superior numerical accuracy compared with previous Allen-Cahn type of LB models. Several benchmark two-phase problems, including static droplet, layered Poiseuille ﬂow, and spinodal decomposition are simulated to validate the present LB model. It is found that the present model can achieve relatively small spurious velocity in the LB community, and the obtained numerical results also show good agreement with the analytical solutions or some available results. Lastly, we use the present model to investigate the droplet impact on a thin liquid ﬁlm with a large density ratio of 1000 and the Reynolds number ranging from 20 to 500. The fascinating phenomena of droplet splashing is successfully reproduced by the present model and the numerically predicted spreading radius exhibits to obey the power law reported in the literature.

DOI: 10.1103/PhysRevE.97.033309

# I. INTRODUCTION

Two-phase ﬂows are ubiquitous in nature and engineering applications. Numerical modeling of such ﬂows becomes an important complement to experimental studies with the rapid development of computational science, while it may face some certain challenges owing to complex interfacial dynamics involving multiple space and time scales. Physically, the interfacial phenomenon can be recognized as a natural consequence of intermolecular interactions. In this regard, the latticeBoltzmann(LB)method[ 1 – 3 ],basedonthemesoscopic kinetic theory, becomes a suitable candidate to model and simulate two-phase ﬂows.

Over the past three decades, the LB method has received great success in modeling multiphase ﬂuid systems [ 1 – 3 ] and some nonlinear equation systems [ 4 , 5 ]. The reasons behind its success lie in the algorithmic simplicity, nature parallelization, and easy implementation of complex boundary. Additionally, thanks to the kinetic nature, the LB method can deal with the intermolecular interactions in a straightforward manner, which is also regarded as its unique advantage that distinguishes it fromthetraditionalcomputationalﬂuiddynamicsmethods.Up to now, a variety of LB models for multiphase ﬂows have been proposed from different physical pictures, which mainly fall into four categories, including color-gradient model [ 6 ], pseudopotential model [ 7 ], free-energy model [ 8 ], and phase-ﬁeldbased model [ 9 – 13 ]. For the detailed expositions, the readers

* shibc@hust.edu.cn

Most of the previously proposed LB models are only able to handle two-phase flows with small or moderate density contrasts. Generally, the density ratio of liquid and vapor phases is larger than 100, and it even could approach 1000 for a realistic water-air two-phase system. Within this context, to develop a multiphase model that can simulate large-densityratio flows is an attractive topic in the LB community. Inamuro et al. [16] proposed a first LB model based on the free-energy method that can tolerate large density differences. However, they need to solve an additional Poisson equation for the pressure to enforce the incompressible condition, which seems to be complex, and undermines the simplicity of the LB method. In addition, an empirical cutoff value is used to determine fluid density, which could lead to the violation of the mass conservation, like the level set method [17]. The extension of the original pseudopotential model [7] to largedensity-ratio cases was attributed to Yuan and Schaefer [18]. They evaluated the performances of different equations of state in the pseudopotential model and found that a large density ratio can be reached with a suitable choice of equation of state. However, it is noticed that their studies only focus on the stationary two-phase problems, and the model will suffer from some limitations more or less when it is readily applied to dynamic two-phase problems. To remove this limitation, Li et al. [19] presented an improved pseudopotential model that can satisfy thermodynamic consistency. Meanwhile it can improve numerical stability of the pseudopotential method at a large density ratio for dynamic flows, which was demonstrated by the simulation of a droplet splashing on a liquid film with the largest density ratio of 700. But they also declared in the literature that it would induce numerical instability when the density ratio was increased to 1000 in this case. Ba et al. [20] also developed a color-gradient-based LB model for simulating two-phase flows with a high density ratio, in which a modified equilibrium distribution function and a simple source term are introduced. They significantly improved the performance of the color-gradient method and achieved satisfactory results in the simulation of droplet splashing problem with the largest density ratio of 100. On the other hand, several researchers have also attempted to develop a large-density-ratio LB model basedonthephase-field theory, which has become increasingly popular in modeling multiphase flows [21]. Zheng et al. [22] proposed a LB two-phase model based on the Cahn-Hilliard phase-field equation and claimed that their model can simulate large-density-ratio flows. Actually, it is noted that they only consider the Navier-Stokes equations on the average density of binary fluids instead of the real fluid density, and therefore their model in theory is only able to deal with density-matched binary fluids, which is also numerically proved by Fakhari and Rahimian [23]. Lee et al. [10,11] also presented another LB model for large-density-ratio two-phase flows from the phasefield viewpoint. The key point of their model in achieving a large density ratio is the use of a stable mixing difference scheme for computing gradient terms, which unfortunately could lead to the violation of mass and momentum conservation [24]. Besides, an inconsistency between the recovered interfacial equation and the target equation in their models was also found [12,25]. Wang et al. [26] proposed an interesting LBflux model for two-phase flows with large density ratios, in which a stable high-order weighted essentially non-oscillator difference scheme is used to solve the Cahn-Hilliard equation, and a like finite volume method for particle distribution function is utilized to solve the incompressible Navier-Stokes equations. Recently, Ren et al. [27] proposed a LB model from the perspective of the Allen-Cahn phase-field equation, while they only concentrated on two-phase flows limited to small or moderate density ratios, and whether it can be applicable for large-density-ratio flows has not been discussed. In addition, their model contains many complex gradient terms, which seem to be implemented with difficulty. More recently, Fakhari and Bolster [28] developed a simple LB model based on the Allen-Cahn phase-field equation that can simulate large-density-ratio two-phase flows. This model utilized a LB equation proposed by Geier et al. [29] to track the interface, which is found to contain some artificial terms in the recovered interfacial equation [27,30]. Therefore the model of Fakhari and Bolster [28] will inherit this weakness in terms of interface capturing, which may affect the numerical accuracy in solving two-phase flows.

In this paper, we intend to present a simple, accurate, and alsorobusttwo-phasemodelforlarge-density-ratioﬂowsinthe framework of the LB method. The proposed LB model is based on the Allen-Cahn phase-ﬁeld theory, which only contains at most a second-order gradient term. Therefore the present model can achieve a high numerical accuracy in interface tracking, compared with the previous LB models [ 9 – 12 , 23 , 25 ] based on the four-order Cahn-Hilliard equation. In addition, a force distribution function is introduced in this model, which can be much simpler than those of the existing LB models

[ 10 – 12 , 25 , 27 , 28 ]. The inconsistency of the recovered interfacial equation with the target equation in Fakhari’s model [ 28 ] is also remedied in this model by the incorporation of a proper source term [ 30 ]. Through the ChapmanEnskog analyis, our model can recover both the conservative Allen-Cahn and the incompressible Navier-Stokes equations correctly, which can be demonstrated to be more accurate than all the previous Allen-Cahn based LB models [ 27 , 28 ]. The rest of the paper is arranged as follows. In Sec. II , the macroscopic governing equations are ﬁrst given, and a LB modelfortwo-phaseﬂowsbasedontheAllen-Cahnphase-ﬁeld theoryisthenpresented. Numericalexperiments tovalidatethe present model and a detailed comparison with some previous LB models can be found in Sec. III , and ﬁnally we made a brief summary in Sec. IV .

# II. LB MODEL FOR TWO-PHASE FLOWS

In this section, we ﬁrst give a brief introduction on the governing equations in the framework of the Allen-Cahn phase-ﬁeld theory [ 31 , 32 ], and then present a LB model for two-phase incompressible ﬂows. Based on the collision operator used, the LB method can be roughly divided into three categories: the single-relaxation-time or so-called BGK method [ 33 ], the two-relaxation-time method [ 34 ], and the multiple-relaxation-time (MRT) method [ 35 ]. Considering its simplicity and high computational efﬁciency, the present model is constructed based on the BGK collision operator and its extension to the advanced MRT version can be conducted directly, which constitutes one of our future research branches.

# A. Governing equations

The conservative Allen-Cahn equation can be expressed by [ 31 , 32 ]

$$
\frac { \partial \phi } { \partial t } + \nabla \cdot ( \phi \mathbf u ) = \nabla \cdot [ M ( \nabla \phi - \lambda \mathbf n ) ] , \quad ( 1 )
$$

where M is the mobility, n is the unit vector normal to the interface,

$$
\mathbf n = \frac { \nabla \phi } { | \nabla \phi | } , \\ \text {of } \phi \text { defined by }
$$

and λ is a function of φ deﬁned by

$$
\lambda = \frac { 4 \phi ( 1 - \phi ) } { W } ,
$$

where W is the interface thickness, φ taking 1 and 0 represents the liquid and gas phase ﬂuids, respectively, and the interface is marked by the contour level of φ = 0 . 5. Here we consider the incompressible two-phase ﬂows, and the ﬂuid velocity u in Eq. ( 1 ) is governed by the following Navier-Stokes equations with the force [ 36 ],

$$
\nabla \cdot \mathbf u = 0 ,
$$

$$
\frac { \partial ( \rho u ) } { \partial t } + \nabla \cdot ( \rho u ) & = - \nabla p + \nabla \cdot [ \mu ( \nabla u + \nabla u ^ { T } ) ] \\ & + F _ { s } + G , \\ \text {where } \rho \text { is the fluid density, } \rho \text { is the hydrodynamic pressure}
$$

where ρ is the fluid density, p is the hydrodynamic pressure, μ is the dynamic viscosity by μ = ρν, ν is the kinematic viscosity, F s is the surface tension force, and G is the possible body force. In the literature [37], there exist several different forms of the surface tension force; here we choose the widely used one of the potential form in the phase-field methods [12,21,25,38],

$$
F _ { s } = \mu _ { \phi } \nabla \phi , & & ( 5 ) & \ n c { 0 } { 5 }
$$

where μ φ is the chemical potential deﬁned by

$$
\mu _ { \phi } = 4 \beta \phi ( \phi - 1 ) ( \phi - 0 . 5 ) - k \nabla ^ { 2 } \phi ,
$$

where k and β are physical parameters that depend on the interface thickness and the surface tension ( σ ),

$$
k = \frac { 3 } { 2 } \sigma W , \ \beta = \frac { 1 2 \sigma } { W } . \quad \quad \quad ( 7 ) \quad \begin{matrix} \text {conf} \\ \text {app} \end{matrix}
$$

# B. LB model for the conservative Allen-Cahn equation

The LB evolution equation with the BGK collision operator for the conservative Allen-Cahn equation can be written

$$
\text {eral} & & f _ { i } ( x + c _ { i } \delta _ { t } , t + \delta _ { t } ) - f _ { i } ( x , t ) = - \frac { 1 } { \tau _ { f } } [ f _ { i } ( x , t ) - f _ { i } ^ { q } ( x , t ) ] \\ & & + \delta _ { t } F _ { i } ( x , t ) , \\ & & \text {where } f _ { i } ( x , t ) \text { is the particle distribution function, } \tau _ { f } \text { is the }
$$

where f i ( x ,t ) is the particle distribution function, τ f is the nondimensional relaxation time related to the mobility, F i ( x ,t ) is the source term, and a simple form of the equilibrium distribution function f eq i ( x ,t ) is given by

$$
f _ { i } ^ { \text {eq} } = \omega _ { i } \phi \left ( 1 + \frac { \text {c} _ { i } \cdot \text {u} } { c _ { s } ^ { 2 } } \right ) , \\ \text {the sound speed} \text { } c _ { s } \text { are the discrete velocities} \text { and }
$$

where c s is the sound speed, c i are the discrete velocities, and ω i are the weighting coefﬁcients. c i and ω i depend on the choice of the lattice model. For the two-dimensional ﬂows considered here, the D2Q5 or D2Q9 lattice model can be applied in the LB algorithm for the Allen-Cahn equation. Considering the consistency with the LB algorithm for NavierStokes equations, in this work we adopt the popular D2Q9 lattice model [ 12 , 13 , 33 , 39 ]. Then the weighting coefﬁcients ω i can be given by ω 0 = 4 / 9, ω 1 − 4 = 1 / 9 , ω 5 − 8 = 1 / 36, and the discrete velocities c i are

$$
\mathfrak { c } _ { i } & = \begin{cases} ( 0 , 0 ) c , & i = 0 , \\ \cos ( [ i - 1 ) \pi / 2 ] , \sin ( [ i - 1 ) \pi / 2 ) c , & i = 1 - 4 , \\ \sqrt { 2 } ( \cos ( [ i - 5 ) \pi / 2 + \pi / 4 ] , \sin [ ( i - 5 ) \pi / 2 + \pi / 4 ] ) c , & i = 5 - 8 , \end{cases}
$$

where c = δ x /δ t is the lattice speed with δ x and δ t representing the grid spacing and the time increment, respectively, and c s = c/ √ 3. By convention, δ x and δ t are set as the length and time units, i.e., δ x = δ t = 1. To recover the Allen-Cahn equation exactly with the multi-

scale analysis, the source term F i in Eq. ( 8 ) should be deﬁned as [ 30 ]

$$
F _ { i } = \left ( 1 - \frac { 1 } { 2 \tau _ { f } } \right ) \frac { \omega _ { i } c _ { i } \cdot \left [ \partial _ { t } ( \phi u ) + c _ { s } ^ { 2 } \lambda \right ] } { c _ { s } ^ { 2 } } , \quad ( 1 1 ) \quad N a v i g h s c r { H }
$$

where the time derivative term ∂ t ( φ u ) is introduced to eliminate the artiﬁcial term in the recovered equation, which is similar to the technique used in LB models [ 12 , 13 , 38 ] for the Cahn-Hilliard equation. One notices that in the existing LB model [ 28 ] based on the Allen-Cahn theory, the term ∂ t ( φ u ) is not included, which results in the deviation between the recovered equation and the target equation [ 27 , 30 ].

The order parameter in the present model can be computed by

$$
\phi = \sum _ { i } f _ { i } . \\ \intertext { f l u i d \, d e n s i y \, i n \, a \, t w o \, - \, n b h s e }
$$

The distribution of ﬂuid density in a two-phase system physically is consistent with that of the order parameter. To satisfy this physical property, the ﬂuid density should take the linear interpolation,

$$
\rho = \phi ( \rho _ { l } - \rho _ { g } ) + \rho _ { g } , \quad & \quad ( 1 3 ) \\ \intertext { l o f } \quad & \quad ( 1 3 ) \quad \text {For} \\
$$

where ρl and ρg represent the densities of the liquid and gas phases. Following the Chapman-Enskog analysis in Ref. [30], it is found that the conservative Allen-Cahn equation can be recovered correctly from the LB equation (8) and the mobility can be determined by

$$
M = c _ { s } ^ { 2 } ( \tau _ { f } - 0 . 5 ) \delta t .
$$

# C. LB model for the Navier-Stokes equations

The LB equation with the BGK collision operator for the Navier-Stokes equations can be expressed as [ 40 , 41 ]

$$
\dim \text {-} \quad g _ { i } ( x + c _ { i } \delta _ { t } , t + \delta _ { t } ) - g _ { i } ( x , t ) = - \frac { 1 } { \tau _ { g } } [ g _ { i } ( x , t ) - g _ { i } ^ { q } ( x , t ) ] \\ \text {in} \quad \text {the} \quad & + \delta _ { t } G _ { i } ( x , t ) , \\ \text {LB} \quad \text {where } g _ { i } ( x , t ) \text { is the distribution function for solving the flow}
$$

where g i ( x ,t ) is the distribution function for solving the ﬂow ﬁeld, g eq i ( x ,t ) is its corresponding equilibrium distribution function, τ g is the dimensionless relaxation time related to the viscosity, and G i ( x ,t ) is the force distribution function. To satisfy the divergence-free condition of velocity, g eq i ( x ,t ) should be elaborately designed as [ 12 , 38 ]

with

$$
g _ { i } ^ { \text {eq} } = \begin{cases} \frac { p } { c _ { s } ^ { 2 } } ( \omega _ { i } - 1 ) + \rho s _ { i } ( \mathbf u ) , & i = 0 , \\ \frac { p } { c _ { s } ^ { 2 } } \omega _ { i } + \rho s _ { i } ( \mathbf u ) , & i \neq 0 \end{cases}
$$

$$
s _ { i } ( u ) = \omega _ { i } \left [ \frac { c _ { i } \cdot u } { c _ { s } ^ { 2 } } + \frac { ( c _ { i } \cdot u ) ^ { 2 } } { 2 c _ { s } ^ { 4 } } - \frac { u \cdot u } { 2 c _ { s } ^ { 2 } } \right ] .
$$

For the two-dimensional ﬂows, the D2Q9 lattice model is also adoptedforﬂowﬁeldandtherelatedphysicalcoefﬁcients ω i , c i are also chosen as those of the previous section.

Different from the previous LB models [ 10 – 12 , 25 , 27 , 28 ], a force distribution function is given by

$$
a \text { once} \, d \text { a} \, u \text { on } \text { time} \, s \text { given by} \\ G _ { i } = \left ( 1 - \frac { 1 } { 2 \tau _ { g } } \right ) \omega _ { i } \left [ u \cdot \nabla \rho + \frac { c _ { i } \cdot F } { c _ { s } ^ { 2 } } + \frac { u \nabla \rho \cdot ( c _ { i } \, c _ { i } - c _ { s } ^ { 2 } ) } { c _ { s } ^ { 2 } } \right ] , \\ \\ \intertext { g h r e f i n s t a l f r a g }
$$

where F is the total force,

$$
F = F _ { s } + G . \intertext { f } \quad \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \
$$

We would like to point out that the force distribution function given in Eq. ( 18 ) can be much simpler than those of the previous models [ 10 – 12 , 25 , 27 , 28 ]. In addition, the present LB model with the force term ( 18 ) can recover the Navier-Stokes equations correctly using the Chapman-Enskog analysis (see the Appendix for the details). Substituting Eqs. ( 5 ) and ( 13 ) into Eq. ( 18 ), one can further simplify Eq. ( 18 ) as

$$
m \text { Eq.} ( 1 6 ) , \text { one can further simply } \text { Eq.} ( 1 6 ) \text { as } & & \text {simple} & & \text { the term} & & \text { the term} & & \text { the term} & & \text { the term} \\ G _ { i } = & \left ( 1 - \frac { 1 } { 2 \tau _ { g } } \right ) & & \phi ( t - 1 ) & & \text {erasure} & & \text {erasure} & & \text {erasure} \\ & & \times \omega _ { i } \left [ \frac { c _ { i } \cdot ( \mu _ { \phi } \nabla \phi + G ) } { c _ { s } ^ { 2 } } + \frac { ( \rho _ { l } - \rho _ { g } ) u \nabla \phi \colon c _ { i } c _ { i } } { c _ { s } ^ { 2 } } \right ] . & & ( 2 0 ) & & \text {second} \\ \text {Taking the zeroth- and the first-order moments of the distrib-}
$$

Taking the zerothand the ﬁrst-order moments of the distribution function g i , the macroscopic quantities u and p can be evaluated as [ 12 , 38 ]

$$
\rho \mathbf u = \sum _ { i } \mathbf c _ { i } g _ { i } + 0 . 5 \delta _ { t } F , & & ( 2 1 a ) \\
$$

$$
p & = \frac { c _ { s } ^ { 2 } } { ( 1 - \omega _ { 0 } ) } \left [ \sum _ { i \neq 0 } g _ { i } + \frac { \delta _ { t } } { 2 } \, u \cdot \nabla \rho + \rho s _ { 0 } ( u ) \right ] , \quad ( 2 l b ) \quad \text {Ocaca} \\ \text {which can be further recast as} & \quad \text {for the} \quad \text {tions}
$$

which can be further recast as

$$
\mu & = \sum _ { i } \mathfrak { c } _ { i } g _ { i } + 0 . 5 \delta _ { t } ( \mu _ { \phi } \nabla \phi + G ) , & ( 2 \mathbf a ) & f _ { i } \\ p & = \frac { c _ { s } ^ { 2 } } { ( 1 - \omega _ { 0 } ) } \left [ \sum _ { i \neq 0 } g _ { i } + \frac { \delta _ { t } } { 2 } ( \rho _ { l } - \rho _ { g } ) \mathbf u \cdot \nabla \phi + \rho s _ { 0 } ( \mathbf u ) \right ] , \\ & \quad \text {where} \quad \mathbf u \text { and } \mathbf m \text { and } ( 2 ) .
$$

with the substitutions of Eqs. ( 5 ) and ( 13 ). Based on the Chapman-Enskog analysis, the ﬂuid kinematic viscosity can be determined by

$$
\nu = c _ { s } ^ { 2 } ( \tau _ { g } - 0 . 5 ) \delta _ { t } . \\
$$

In a two-phase system, the viscosity is no longer a uniform value due to its jump at the liquid-gas interface. There are several manners in which to treat the viscosity across the interface. To be smooth across the interface, the viscosity in the diffusion-interface methods is usually supposed to be a linear function of the order parameter [ 9 ],

$$
\nu = \phi ( \nu _ { l } - \nu _ { g } ) + \nu _ { g } , \quad & ( 2 4 ) \\ \intertext { v = \phi ( \nu _ { l } - \nu _ { g } ) + \nu _ { g } , } & \quad \intertext { v = \phi ( \nu _ { l } - \nu _ { g } ) + \nu _ { g } , } & \quad \intertext { v = 1 \cdot \cdot \cdot }
$$

where ν l and ν g are the kinematic viscosities of the liquid and gas phases. In addition to Eq. ( 24 ), another popular treatment to determine the viscosity is the inverse linear form [ 11 , 28 ],

$$
\frac { 1 } { \nu } = \phi \left ( \frac { 1 } { \nu _ { l } } - \frac { 1 } { \nu _ { g } } \right ) + \frac { 1 } { \nu _ { g } } .
$$

Oftentimes, to avoid the sharp-interface limit of the phaseﬁeld methods, a step function is also applied for the dynamic viscosity [ 27 ],

$$
\mu = \begin{cases} \mu _ { l } , & \phi \geqslant 0 . 5 , \\ \mu _ { g } , & \phi < 0 . 5 , \end{cases} \\
$$

where μ l and μ g are the dynamic viscosities of two different phases. The scheme ( 26 ) can achieve a considerable accuracy in tracking the interface, while similar to the sharp-interface methods, it could be unstable when it is applied to interfacial dynamic problems with large topology change [ 42 ]. In this work, a simple linear form as used for density is adopted, if not speciﬁed.

For numerical iterations, the derivative terms in the model should be discretized with suitable difference schemes. For simplicity, we adopt the explicit Euler scheme to compute the temporal derivative in Eq. ( 11 ), i.e., ∂ t ( φ u ) = [ φ ( t ) u ( t ) − φ ( t − δ t ) u ( t − δ t )] /δ t [ 4 , 12 ]. As commonly used in LB literatures [ 12 , 13 , 25 , 38 ], the gradient term is computed by the second-order isotropic central scheme,

$$
\nabla \phi ( x ) = \sum _ { i \neq 0 } \frac { \omega _ { i } \mathfrak { c } _ { i } \phi ( x + \mathfrak { c } _ { k } \delta _ { t } ) } { c _ { s } ^ { 2 } \delta _ { t } } \\
$$

and the Laplace operator is calculated by

$$
\nabla ^ { 2 } \phi ( x ) = \sum _ { i \neq 0 } \frac { 2 \omega _ { i } [ \phi ( x + \mathfrak { c } _ { i } \delta _ { t } ) - \phi ( x ) ] } { c _ { s } ^ { 2 } \delta _ { t } ^ { 2 } } . \\
$$

Occasionally, the gradient term can be computed with the nonequilibrium part in some certain LB approaches [ 5 , 30 ] for the convection-diffusion equations. To derive the computational scheme, we ﬁrst introduce the multiscale expansions [ 2 ],

$$
f _ { i } & = f _ { i } ^ { ( 0 ) } + \epsilon f _ { i } ^ { ( 1 ) } + \epsilon ^ { 2 } f _ { i } ^ { ( 2 ) } + \cdots , \\ \partial _ { t } & = \epsilon \partial _ { t _ { 1 } } + \epsilon ^ { 2 } \partial _ { t _ { 2 } } , \quad \nabla = \epsilon \nabla _ { 1 } , \quad F _ { i } = \epsilon F _ { i } ^ { ( 1 ) } + \epsilon ^ { 2 } F _ { i } ^ { ( 2 ) } ,
$$

where   is a small parameter. Applying the Taylor expansion and multiscale formulas to Eq. ( 8 ), one can derive the consecutive equations in   ,

$$
D _ { 1 i } f _ { i } ^ { ( 0 ) } = - \frac { 1 } { \tau _ { f } \delta _ { t } } f _ { i } ^ { ( 1 ) } + F _ { i } ^ { ( 1 ) } ,
$$

$$
\partial _ { t _ { 2 } } f _ { i } ^ { ( 0 ) } + & \left ( 1 - \frac { 1 } { 2 \tau _ { f } } \right ) D _ { 1 i } f _ { i } ^ { ( 1 ) } + \frac { \delta _ { t } } { 2 } D _ { 1 i } F _ { i } ^ { ( 1 ) } \\ = & - \frac { 1 } { \tau _ { f } \delta _ { t } } f _ { i } ^ { ( 2 ) } + F _ { i } ^ { ( 2 ) } ,
$$

where D 1 i = ∂ t 1 + c i · ∇ 1 . From Eq. ( 30a ), one can easily obtain the ﬁrst-order moment of f (1) i ,

$$
\sum _ { i } c _ { i } f _ { i } ^ { ( 1 ) } & = - \tau _ { f } \delta _ { t } \left [ c _ { s } ^ { 2 } \nabla _ { 1 } \phi + \frac { 1 } { 2 \tau _ { f } } \partial _ { t _ { 1 } } ( \phi \mathbf u ) \\ & - \left ( 1 - \frac { 1 } { 2 \tau _ { f } } \right ) c _ { s } ^ { 2 } \lambda \mathbf n ^ { ( 1 ) } \right ] . \\
$$

Multiplying   on both sides of Eq. ( 31 ), we can rewrite Eq. ( 31 ) as

$$
\sum _ { i } \mathfrak { c } _ { i } ( f _ { i } - f _ { i } ^ { ( e q ) } ) + \frac { \delta _ { t } } { 2 } \partial _ { t } ( \phi \mathfrak { u } ) & & \text {the} \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \
$$

where the approximation  f (1) i ≈ f i − f eq i has been applied, and the time derivative term at t 2 diffusion scale has been neglected. After some algebraic manipulations, one can ultimately derive the computational schemes for the gradient of the phase-ﬁeld variable and its gradient norm [ 30 ]

$$
| \nabla \phi | = \frac { - | C | - B } { A } , \quad \quad ( 3 3 a ) \quad \ e l s \\
$$

$$
\nabla \phi = \frac { C } { A + \frac { B } { | \nabla \phi | } } , \quad \quad ( 3 3 b ) \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \
$$

where A = − c 2 s τ f δ t , B = Mδ t λ , and C =   i c i ( f i − f eq i ) + 0 . 5 δ t ∂ t ( φ u ).FromEqs.( 33a )and( 33b ),wecanclearlyobserve that the gradient can be calculated locally from the nonequilibrium part of the distribution function without any ﬁnite differences involved. In the following, we intend to present a discussion on the accuracy of the nonequilibrium scheme, which is not given in Ref. [ 30 ]. Obviously, the error arises from the estimate of C , where the high-order terms of the nonequilibrium part ( f ( k ) i ,k   2) and the term − δ t 2   2 ∂ t 2 ( φ u ) have been removed. We only concentrate on the effect of the second-order term of the nonequilibrium part, since its higher-order terms are of little importance with the increasing order of   [ 2 , 43 ]. In this case, we can derive the expression of the main truncation error term,

$$
\sum _ { i } \mathfrak { c } _ { i } f _ { i } ^ { ( 2 ) } = - \tau _ { f } \delta _ { t } \left [ \frac { 1 } { 2 \tau _ { f } } \partial _ { t _ { 2 } } ( \phi \mathfrak { u } ) - M \partial _ { t _ { 1 } } ( 2 \nabla _ { 1 } \phi - \lambda \mathfrak { n } ^ { ( 1 ) } ) & & \text {and the} \\ & & \text {and} \\ & & - M \nabla _ { 1 } \cdot ( \nabla _ { 1 } \cdot \phi \Delta \cdot \mathfrak { u } ) \right ] , \\ \text {where} \, & & \text {and} \, \text {according to Refs. [43,44]}, \text {it} & & \text {all}
$$

where Eq. ( 30b ) has been used. According to Refs. [ 43 , 44 ], it is known that the lattice spacing and time interval satisfy the relation δ t ∝ δ 2 x atthediffusionscale,andconsequently,wecan ﬁnd   i c i f (2) i ∝ δ 2 x and δ t 2 ∂ t 2 ( φ u ) ∝ δ 2 x . Based on the above results, we can derive the relation C =   i c i ( f i − f eq i ) + 0 . 5 δ t ∂ t ( φ u ) + O ( δ 2 x ), which indicates that the nonequilibrium scheme can also achieve a second-order accuracy in space as the ﬁnite difference scheme. In practice, when the | ∇ φ | and ∇ φ are computed by Eqs. ( 33a ) and ( 33b ), the unit normal vector n can then be obtained. This treatment on the gradient termenablesthecollisionprocesstobeimplementedlocally,in addition to the computation of μ φ , which is one of the striking features in LB approaches. Therefore the scheme will be adopted in our simulations, unless otherwise stated. However, we ﬁnd that the velocity actually satisﬁes an implicit equation, when Eqs. ( 33a ) and ( 33b ) are applied for the statistics of the velocity, and for simplicity we just applied Eqs. ( 27 ) and ( 28 ) in this step.

At the end of this section, we would like to give some remarks on the present model for two-phase ﬂows. Firstly, the present model is developed based on the conservative Allen-Cahn equation, which contains a lower-order diffusion term compared with the fourth-order Cahn-Hilliard equation. From the theoretical point of view, the Cahn-Hilliard equation cannot be directly recovered from the LB models through the second-order Chapman-Enskog analysis. Therefore the Allen-Cahn based model in theory can achieve a higher numerical accuracy in solving the index function φ and also the density ﬁeld via Eq. ( 13 ) than the Cahn-Hilliard type of LB models. The higher-precision solution of φ plays a signiﬁcant role in simulating large-density-ratio ﬂows since a small deviation could be more likely to lead an unphysically negative value of ﬂuid density, causing numerical instabilities. It is worth noting that a type of large-density-ratio LB models [ 10 , 11 ] have been proposed based on the Cahn-Hilliard equation, which is attributed to the use of a mixed scheme that combines the central and biased differences. However, it will induce the violations of mass and momentum conservation [ 24 ]. On the contrary, in our model the isotropic central scheme and the local nonequilibrium scheme are applied, which not only preserve a second-order accuracy in space, but also can ensure the global mass conservation of a two-phase system. Secondly, a force distribution function for ﬂow ﬁeld is proposedinthepresentmodel,whichcanbemuchsimplerthan those of the existing Allen-Cahn based LB models [ 27 , 28 ]. It is noted that our model only contains one type of nonlocal gradient term for the order parameter, which is much smaller than those of the previous model [ 27 ]. In addition, the gradient term and its modulus in our model can be computed with local nonequilibrium schemes, which enables the collision process to be conducted locally if μ φ has been given. Whereas, in the previous models [ 27 , 28 ], only the central difference schemes are applied, and thus the collision process cannot be conducted locally. Thirdly, both the conservative Allen-Cahn equation and the incompressible Navier-Stokes equations can be recovered exactly from the present model with the multiscale analysis. Whereas, the model of Fakhari and Bolster [ 28 ] contains some artiﬁcial terms in the recovered interfacial equation. The numerical experiments conducted below will demonstrate that the present model is more accurate than the previous Allen-Cahn based LB models [ 27 , 28 ]. Lastly, we would like to stress that the present model is a standard LB scheme for simulating large-density-ratio two-phase ﬂows without the use of an advanced ﬁnite difference or ﬁnite volume method [ 26 ], therefore it can naturally inherit the advantages of the LB methodindealingwithcomplexphysicalboundaryandparallel computing.

# III. NUMERICAL RESULTS AND DISCUSSIONS

In this section, several typical benchmark problems, including static droplet, layered Poiseuille ﬂows, and spinodal decomposition are used to validate the present LB model for large-density-ratio ﬂows. We attempt to conduct detailed comparisons between the present results and the analytical solutions or some available results. Lastly, we also investigated droplet impact on a thin liquid ﬁlm, where the effect of the Reynolds number is discussed in detail.

200

![](<liang2018_images/imageFile2.png>)

(a)

1000

(b)

150

800

Analytical solution

M=0.1

600

M=0.05

M=0.02

100

y

ρ

400

50

200

0

0

0

50

100

150

200

0

50

100

150

200

x

x

FIG. 1. Static droplet tests with density ratio ρ l /ρ g = 1000 : 1. (a) The velocity distribution of the whole domain at the equilibrium state. The solid and dashed lines represent the equilibrium shape of the droplet and its initial shape, respectively; (b) the density distributions along the horizontal center line ( y = N y / 2) at different values of the mobility.

# A. Static droplet

The static droplet is a basic two-phase problem, which has been widely used to verify the developed numerical methods [ 10 , 12 , 19 , 20 , 23 , 25 ]. In this section, we will simulate this problem with large density ratio to validate the present LB model. Initially, a liquid droplet with the radius of R = 50 surrounded by the gas phase is located at the center of the squaredomain N y × N x = 200 × 200andtheperiodicboundary conditionsareappliedatallboundaries.Thedistributionproﬁle of the order parameter is initialized by

$$
\phi ( x , y ) = 0 . 5 + 0 . 5 \tanh \frac { 2 [ R - ( x - 1 0 0 ) ^ { 2 } - ( y - 1 0 0 ) ^ { 2 } ] } { W } , \quad \begin{array} { c c } \text {all} & \text {suit} \\ \text {between} & \text {condu} \end{array} , \quad \begin{array} { c c } \text {all} & \text {suit} \\ \text {condu} & \text {previor} \end{array} , \quad \begin{array} { c c } \text {all} & \text {suit} \\ \text {previor} & \text {velocity} \end{array}
$$

which enables its value to be smooth across the interface. In the simulation, we set the density ratio to be ρ l /ρ g = 1000 : 1, and some other physical parameters are given as ν l = ν g = 0 . 1, σ = 0 . 001 , W = 5. Figure 1(a) depicts the interface pattern of the droplet at the equilibrium state, together with the initial one given by Eq. ( 35 ). It can be found that they line up over each other exactly, which indicates that the present model has a high accuracy in tracking the interface. Furthermore, we quantitatively plotted in Fig. 1(b) the density distribution along the horizontal center line with different values of the mobility M . It is shown that numerical predictions of the density ﬁeld are all in good agreement with the analytical solution.

The spurious velocity around the interface is a commonly concerned problem in LB approaches for two-phase flows, and cannot be completely eliminated in the framework of the LB method [45]. In Fig. 1(a), we also display the velocity distribution in the whole computational domain obtained by the present model. It can be found that the spurious velocities indeed exist at the vicinity of the interface, and their maximum magnitude computed by | u | max = ( √ u 2 + v 2 )max has an order of 10 -9 . The effect of the density ratio on the spurious velocities is investigated. We simulated this problem with a wide range of density ratios from 10 to 1000, and the obtained results showed that the spurious velocities at least have the order of 10 -8 . We also examined the effect of the dimensionless Laplace number defined by Rσ/νl on the spurious velocity. Different Laplace numbers are obtained by changing the surface tension coefficient. The numerical experiments indicate that the maximum magnitudes of the spurious velocities for all situations have an order of 10 -9 for the Laplace number between 0.05 and 5. Lastly in this subsection, we further conducted comparisons between the present model and some previously improved LB models in terms of the spurious velocity. It has been reported that the maximum amplitude of spurious velocities in an improved Shan-Chen model [46] has the order of 10 -3 . Recently, Ba et al. [20] developed an improved color-gradient-based model for high density ratio, which produced spurious velocities with the order of 10 -5 . As for the Cahn-Hilliard type of LB model [12], they can obtain spurious velocities at the level of 10 -6 . From the above discussion, it can be concluded that the present LB model is able to produce relatively small spurious velocities.

# B. Layered Poiseuille ﬂow

The layered Poiseuille flow is a classical two-phase problem, which can provide a good benchmark for validating the developed LB approaches [20,25-27,47,48]. To our knowledge, most of the previous studies are limited to the small density ratio less than 10, due to the numerical instability problem. In this section, we will simulate the layered two-phase Poiseuille flows with the largest density ratio of 1000, and also conduct comparisons of the present model with the existing Allen-Cahn based LB models [27,28]. Consider a channel flow of two immiscible fluids driven by a constant body force G = ( Gx, 0). Initially, the gas phase fluid is placed in the upper region of 0 < y ⩽ h and the region of -h ⩽ y ⩽ 0 is filled with the liquid phase fluid. Periodic boundary conditions are applied in the x direction, and the bottom and top boundaries are the solid walls, which are treated by the halfway bounce-back boundary conditions. Based on these boundary conditions, one can derive the analytical solution for the horizontal velocity profile ( ux ),

![](<liang2018_images/imageFile3.png>)

10 -4

10 -4

×

×

2

14

(b)

(a)

Analytical solution Present model

1.8

Analytical solution Present model

12

Present model

Present model

al. [27] Model of Fakhari et al.[28]

al. [27] Model of Fakhari et al. [28]

1.6

Model of Fakhari et al.[28]

Model of Fakhari et al. [28]

10

1.4

8

1.2

x

x

6

1

u

u

0.8

4

0.6

2

0.4

0

0.2

-2

0

-50

0

5

0

-50

0

5

0

y

y

10 -3

10 -4

×

×

2

14

(d)

Analytical solution Present model

Analytical solution Present model

(c)

1.8

12

Present model

Present model

al. [27] Model of Fakhari et al. [28]

al. [27]. Model of Fakhari et al. [28]

1.6

Model of Fakhari et al. [28]

10

Model of Fakhari et al. [28]

1.4

8

1.2

6

x

x

1

u

u

4

0.8

0.6

2

0.4

0

0.2

-2

0

-50

0

5

0

-60

-40

-20

0

20

40

60

y

y

FIG. 2. Comparisons of the horizontal velocity proﬁles obtained by the present model and the existing Allen-Cahn based LB models [ 27 , 28 ] with various density ratios: (a) ρ l : ρ g = 10 : 1; (b) ρ l : ρ g = 100 : 1; (c) ρ l : ρ g = 150 : 1; (d) ρ l : ρ g = 1000 : 1. The solid lines represent the corresponding analytical solutions.

$$
& \quad \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \
$$

where G x = u c ( μ l + μ g ) /h 2 , which provides a steady horizontalvelocityof u c atthecenter.Toquantitativelydescribethe accuracy of the present model and also conveniently compare

$$
E _ { u } = \frac { \sum _ { y } \left | u _ { x } ^ { n } ( y , t ) - u _ { x } ^ { a } ( y ) \right | } { \sum _ { y } \left | u _ { x } ^ { a } ( y ) \right | } , \\ \text {the subscripts } n \text { and } a \text { denote the numerical and } \\ \text {solution}
$$

u =       y   u a x ( y )   where the subscripts n and a denote the numerical and analytical solutions.

In the simulation, the computational grid is chosen to be N y × N x = 100 × 10, and the initial distribution of the order parameter is set as

$$
\phi ( x , y ) = 0 . 5 + 0 . 5 \tanh \frac { 2 ( 0 . 5 N _ { y } - y ) } { W } ,
$$

which gives the proﬁle of the planar interface. u c is ﬁxed as a small value of 10 − 4 , which ensures that the incompressible limit can be satisﬁed, and some other related parameters are given as W = 5 , σ = 0 . 001 , ν l = 0 . 1, and M = 0 . 1. Four different cases of the density ratios ρ l /ρ g = 10 , 100 , 150 , 1000

TABLE I. Relative errors of the horizontal velocity ( u x ) in layered Poiseuille tests.

|Density ratio Present ( )|Present LB model|of Ren et al. [ 27 ]|Model of Fakhari and Bolster [ 28 ]|
|---|---|---|---|
|10|8 . 9 × 10 - 3|1 . 0 × 10 - 2|6 . 2 × 10 - 2|
|100|9 × 10 − 9 × 10 − 3|4 . 4 × 10 - 2|2 . 7 × 10 - 1|
|150|4 × 10 − 3 2|8 . 2 × 10 − 2 1|3 . 0 × 10 − 1 1|
|1000 3 .|2 × 10 −|1 . 1 × 10 −|3 . 9 × 10 −|


are considered, where the corresponding kinematic viscosity ratios ν g /ν l are 1 for the former three cases, and 10 for the latter case. Here the dynamic viscosity is given by Eq. ( 26 ) in the present test of our model. Note that the two-phase system with ρ l /ρ g = 1000 and μ l /μ g = 100 considered here is very close to the realistic water-air system at room temperature and normalatmosphericpressure.Figure 2 showstheproﬁlesofthe horizontal velocity ( u x ) with various density ratios obtained by the present model, together with the corresponding analytical solutions. For comparisons, we also simulated the above cases with the previous Allen-Cahn based LB models [ 27 , 28 ] under identicalcomputationalconditions,andtheobtainednumerical results are also presented in Fig. 2 . It can be observed from Fig. 2 that the numerical results of the present model agree well with the analytical solutions for all density ratios, while some obvious discrepancies with the analytical solutions are found in the results of the existing Allen-Cahn based LB models[ 27 , 28 ],especiallyathighdensityratios.Wealsoconducted a quantitative comparison between the present model and the previous LB models [ 27 , 28 ]. The relative errors of the velocity u x with these LB models were measured and the results are summarized in Table I . It is found that the previous models produce large relative errors, and they all increase signiﬁcantly with the density ratio. In contrast, a much smaller relative error can be derived by the present model, which also seems to be independent of the density ratio. Based on the above discussion, we can see that the present model is more accurate than the previous Allen-Cahn based LB models [ 27 , 28 ].

# C. Spinodal decomposition

Spinodal decomposition [ 49 ] is a fundamental property of a ﬂuid mixture with two different species. For suitable compositions and quenches, the initial homogeneous mixture is unstable in the presence of small ﬂuctuations, and then the spinodal decomposition phenomenon will take place. This phenomenon, ubiquitous in physics and chemistry, has been studied extensively. Several researchers have also investigated the spinodal decomposition problem using the LB approaches [ 7 , 8 , 25 , 50 ],whiletheymainlyfocusontheprocess of phase separation with small or moderate density ratios. In this section, we intend to simulate this problem with the large density ratio of 1000 by the present LB model, where the gradient terms are computed by Eqs. ( 27 ) and ( 28 ). This exercise is devoted to the demonstration of the capability of our method in studying complex high-density-ratio twophase ﬂows. The computational mesh used here is chosen to be N y × N x = 200 × 200. The periodic boundary conditions are applied at all boundaries. In the simulation, the initial distribution of the order parameter with small ﬂuctuations can

$$
\phi ( x , y ) = \frac { 1 } { 3 } + \text {rand} ( x , y ) ,
$$

where rand( x,y ) is a random function with the maximum amplitude of 0.01. Then a small perturbation can be imposed on a homogeneous density ﬁeld via Eq. ( 13 ), where ρ l and ρ g are set to be 1000 and 1. We only consider binary ﬂuids with the viscosity ratio of ν g /ν l = 10, which approaches that of a water-air system. The remaining parameters in the simulation are ﬁxed as σ = 0 . 2 , W = 4, and M = 0 . 1. Figure 3 depicts the time evolution of the density distribution during the phase separating process, where the time ( t ) has been nondimensionalized by the viscous time of the liquid phase ρ l ν l W/σ . It can be found that the early stage of phase separation induces small ﬂuctuations of the density into large-scale inhomogeneities. Then some tiny droplets with random shapes are formed in the system. The droplet sizes increase with time, and some of them also coalesce into the larger ones, which leads to the eventual separation of binary ﬂuid components. The above phase separating processes are results of the hydrodynamics and surface tension action, which conform to the expectation.

# D. Droplet impact on a thin liquid ﬁlm

Lastly, to show the capacity of the present model, we consider a complex problem of droplet impact dynamics with large density ratio. Droplet impact on liquid surfaces [ 51 ] is a familiar spectacle in the natural event of a falling raindrop on the wet ground or a puddle. Further, it plays a prominent role in many technical applications, such as ink jet printing, spray cooling, and and coating. In spite of its ubiquity and extensiveresearch[ 51 – 54 ],numericalsimulationofsuchﬂows still poses some challenges due to complex interfacial changes in topology, and yet there exists a large density difference for a water-air system. In addition, a numerical singularity may be produced at the impact point. In this section, we will simulate a two-dimensional droplet impact on a preexisting thin liquid ﬁlm with a large density ratio of 1000 by the present LB model, in the absence of the gravitational ﬁeld.

The simulations are performed on a uniform computational mesh with the size of L × H = 1500 × 500, as illustrated in Fig. 4 . A wetting liquid ﬁlm with the height of H w = 0 . 1 H is initially located at the bottom wall, and a circular droplet with the radius ( R ) of 100 lattice units is just placed on the upper region of the liquid ﬁlm. In the simulation, the distribution of the order parameter can be initialized by

$$
\phi ( x , y ) = 0 . 5 + 0 . 5 \tanh \frac { 2 ( H _ { w } - y ) } { W } ,
$$

![](<liang2018_images/imageFile2.png>)

(a)

(b

(c)

(d)

(e)

(h)

FIG. 3. Time evolution of the density distribution during the phase separating processes, (a) t = 0; (b) t = 0 . 05; (c) t = 0 . 25; (d) t = 0 . 5; (e) t = 5; (f) t = 25; (g) t = 50; (h) t = 125.

and also

$$
\phi ( x , y ) = 0 . 5 + 0 . 5 \tanh \\ \times \frac { 2 [ R - ( x - 0 . 5 L ) ^ { 2 } - ( y - R - H _ { w } ) ^ { 2 } ] } { W } , \ y > H _ { w } , \quad \text {and}
$$

where W is the interface thickness and is set to be 5. The velocity ﬁeld at the initial time can be assigned by

$$
( u , v ) = \begin{cases} ( 0 , - \phi U ) , & y > H _ { w } , \\ ( 0 , 0 ) , & y \leq H _ { w } , \end{cases} \quad \begin{array} { c c c } ( 4 2 ) & & ( 4 2 ) \\ & & \\ & & \\ \end{array} \quad \begin{array} { c c c } ( 4 2 ) & & \\ & & \\ & & \\ \end{array} \quad \begin{array} { c c c } ( 4 2 ) & & \\ & & \\ & & \\ \end{array} \quad \begin{array} { c c c } ( 4 2 ) & & \\ & & \\ & & \\ \end{array} \quad \begin{array} { c c c } ( 4 2 ) & & \\ & & \\ & & \\ \end{array} \quad \begin{array} { c c c } ( 4 2 ) & & \\ & & \\ & & \\ \end{array} \quad \begin{array} { c c c } ( 4 2 ) & & \\ & & \\ & & \\ \end{array} \quad \begin{array} { c c c } ( 4 2 ) & & \\ & & \\ & & \\ \end{array} \quad \begin{array} { c c c } ( 4 2 ) & & \\ & & \\ & & \\ \end{array} \quad \begin{array} { c c c } ( 4 2 ) & & \\ & & \\ & & \\ \end{array} \quad \begin{array} { c c c } ( 4 2 ) & & \\ & & \\ & & \\ \end{array} \quad \begin{array} { c c c } ( 4 2 ) & & \\ & & \\ & & \\ \end{array} \quad \begin{array} { c c c } ( 4 2 ) & & \\ & & \\ & & \\ \end{array} \quad \begin{array} { c c c } ( 4 2 ) & & \\ & & \\ & & \\ \end{array} \quad \begin{array} { c c c } ( 4 2 ) & & \\ & & \\ & & \\ \end{array} \quad \begin{array} { c c c } ( 4 2 ) & & \\ & & \\ & & \\ \end{array} \quad \begin{array} { c c c } ( 4 2 ) & & \\ & & \\ & & \\ \end{array} \quad \begin{array} { c c c } ( 4 2 ) & & \\ & & \\ & & \\ \end{array} \quad \begin{array} { c c c } ( 4 2 ) & & \\ & & \\ & & \\ \end{array} \quad \begin{array} { c c c } ( 4 2 ) & & \\ & & \\ & & \\ \end{array} \quad \begin{array} { c c c } ( 4 2 ) & & \\ & & \\ & & \\ \end{array} \quad \begin{array} { c c c } ( 4 2 ) & & \\ & & \\ & & \\ \end{array} \quad \begin{array} { c c c } ( 4 2 ) & & \\ & & \\ & & \\ \end{array} \quad \begin{array} { c c c } ( 4 2 ) & & \\ & & \\ & & \\ \end{array} \quad \begin{array} { c c c } ( 4 2 ) & & \\ & & \\ & & \\ \end{array} \quad \begin{array} { c c c } ( 4 2 ) & & \\ & & \\ & & \\ \end{array} \quad \begin{array} { c c c } ( 4 2 ) & & \\ & & \\ & & \\ \end{array} \quad \begin{array} { c c c } ( 4 2 ) & & \\ & & \\ & & \\ \end{array} \quad \begin{array} { c c c } ( 4 2 ) & & \\ & & \\ & & \\ \end{array} \quad \begin{array} { c c c } ( 4 2 ) & & \\ & & \\ & & \\ \end{array} \quad \begin{array} { c c c } ( 4 2 ) & & \\ & & \\ & & \\ \end{array} \quad \begin{array} { c c c } ( 4 2 ) & & \\ & & \\ & & \\ \end{array} \quad \begin{array} { c c c } ( 4 2 ) & & \\ & & \\ & & \\ \end{array} \quad \begin{array} { c c c } ( 4 2 ) & & \\ & & \\ & & \\ \end{array} \quad \begin{array} { c c c } ( 4 2 ) & & \\ & & \\ & & \\ \end{array} \quad \begin{array} { c c c } ( 4 2 ) & & \\ & & \\ & & \\ \end{array} \quad \begin{array} { c c c } ( 4 2 ) & & \\ & & \\ & & \\ \end{array} \quad \begin{array} { c c c } ( 4 2 ) & & \\ & & \\ & & \\ \end{array} \quad \begin{array} { c c c } ( 4 2 ) & & \\ & & \\ & & \\ \end{array} \quad \begin{array} { c c c } ( 4 2 ) & & \\ & & \\ & & \\ \end{array} \quad \begin{array} { c c c } ( 4 2 ) & & \\ & & \\ & & \\ \end{array} \quad \begin{array} { c c c } ( 4 2 ) & & \\ & & \\ & & \\ \end{array} \quad \begin{array} { c c c } ( 4 2 ) & & \\ & & \\ & & \\ \end{array} \quad \begin{array} { c c c } ( 4 2 ) & & \\ & & \\ & & \\ \end{array} \quad \begin{array} { c c c } ( 4 2 ) & & \\ & & \\ & & \\ \end{array} \quad \begin{array} { c c c } ( 4 2 ) & & \\ & & \\ & & \\ \end{array} \quad \begin{array} { c c c } ( 4 2 ) & & \\ & & \\ & & \\ \end{array} \quad \begin{array} { c c c } ( 4 2 ) & & \\ & & \\ & & \\ \end{array} \quad \begin{array} { c c c } ( 4 2 ) & & \\ & & \\ & & \\ \end{array} \quad \begin{array} { c c c } ( 4 2 ) & & \\ & & \\ & & \\ \end{array} \quad \begin{array} { c c c } ( 4 2 ) & & \\ & & \\ & & \\ \end{array} \quad \begin{array} { c c c } ( 4 2 ) & & \\ & & \\ & & \\ \end{array} \quad \begin{array} { c c c } ( 4 2 ) & & \\ & & \\ & & \\ \end{array} \quad \begin{array} { c c c } ( 4 2 ) & & \\ & & \\ & & \\ \end{array} \quad \begin{array} { c c c } ( 4 2 ) & & \\ & & \\ & & \\ \end{array} \quad \begin{array} { c c c } ( 4 2 ) & & \\ & & \\ & & \\ \end{array} \quad \begin{array} { c c c } ( 4 2 ) & & \\ & & \\ & & \\ \end{array} \quad \begin{array} { c c c } ( 4 2 ) & & \\ & & \\ & & \\ \end{array} \quad \begin{array} { c c c } ( 4 2 ) & & \\ & & \\ & & \\ \end{array} \quad \begin{array} { c c c } ( 4 2 ) & & \\ & & \\ & & \\ \end{array} \quad \begin{array} { c c c } ( 4 2 ) & & \\ & & \\ & & \\ \end{array} \quad \begin{array} { c c c } ( 4 2 ) & & \\ & &
$$

where U is the impact velocity with a ﬁxed value of 0.05. The periodic boundary conditions are applied at the left and right boundaries, while the no-slip bounce-back boundary condition is imposed at the bottom wall and the open boundary condition is utilized at the top boundary. Two major dimensionless parameters governing droplet impact are the Reynolds number

![](<liang2018_images/imageFile3.png>)

Vapor

H

FIG. 4. Schematic of the initial setup for the droplet impact on a thin liquid ﬁlm.

$$
R e = \frac { \rho _ { l } D U } { \mu _ { l } }
$$

and Thenthedropletcontinuestospread,followedbytheformation of the ejecta sheet at the intersection region between the droplet and the liquid layer. The ejecta sheet grows into a splashing lamella [known as crown in axisymmetric or three-dimensional (3D) geometry] propagating radially with increasing time and tends to bend at its end rim. The splashing phenomenon is also observed in a moderate Re of 100, as shown in Fig. 6, while the extent is significantly reduced. This is ascribed to the increasing frictional force between the liquid phase and its ambient vapor phase at a larger viscosity, and then the interface layer can be more stable. As the Reynolds number is lowered to a small value of 20, we do not observe the splashing behavior of the droplet in Fig. 7, as expected. The droplet only merges with the thin liquid film, which evolves in a manner of the outward moving surface wave. This process of droplet impact is oftentimes named as deposition, which is in line with the results of the previous studies [10,19].

$$
W e = \frac { \rho _ { l } D U ^ { 2 } } { \sigma } ,
$$

where D is the droplet diameter. The Weber number has been taken ﬁxed and equals We = 8000, as commonly used in other studies [ 10 , 52 , 55 ]. The density ratio of the liquid and gas phases is set to ρ l /ρ g = 1000:1. Three typical Reynolds numbers Re = 500, 100, and 20 are considered in this work, which are derived by tuning the kinematic viscosity of the liquid phase while keeping the gas kinematic viscosity as a constant. With this strategy, it is found that the lowest achievable liquid kinematic viscosity is 0.02 at the largest Re of 500, and the viscosity ratio ( ν g /ν l ) at this situation is 10, which is very close to that of a realistic water-air two-phase system. With the driving of the impact velocity, the system is released and the droplet will instantly impact onto the underneath ﬁlm. Here we mainly concentrate on the interfacial dynamics and the variation law of the spreading radius versus time. Figures 5 – 7 depict typical scenic representations of the droplet impact process at three different Reynolds numbers of 20, 100, and 500, where the time instants t ∗ is the normalized time deﬁned by t ∗ = tU/D ; t is the iteration step. For high Re of 500, the droplet moves downward instantly with slight deformation at the initial stage, and some tiny bubble-ring entrapments are visible in the neck connecting the droplet and ﬁlm. The bubble entrapment phenomenon has also been reported in the recent studies on the droplet impact [ 53 , 54 ].

![](<liang2018_images/imageFile6.png>)

FIG. 5. Snapshots of droplet impact on a thin liquid ﬁlm with Re = 500 , We = 8000, and ρ l /ρ g = 1000. The time instants t ∗ have been normalized by the characteristic time D/U .

FIG. 6. Snapshots of droplet impact on a thin liquid ﬁlm with Re = 100 , We = 8000, and ρ l /ρ g = 1000. The time instants t ∗ have been normalized by the characteristic time D/U .

We also conducted a quantitative study on the spreading radius, which is a concerning physical quantity in droplet impact dynamics [ 51 ]. Previous research [ 10 , 19 , 20 , 52 , 55 ] has indicated that the growth of the spreading radius generally can be described by the power law r/D = C √ Ut/D , where C is a coefﬁcient that depends on the ﬂow geometry. For the axisymmetric or 3D modeling of the droplet impact, the coefﬁcient C isfoundtobe1.1byJosserandaandZaleskib[ 55 ]. Whereas, for the plane two-dimensional situation, the scaled prefactor C is found to be larger than 1.1, as reported in several literatures [ 10 , 15 , 20 , 52 ]. Figure 8 shows the time variation of the numerically predicted spreading radius by the present model. For a comparison, the theoretical result of the ﬁtting power formula is also presented. The comparison between them shows a good agreement in general, except for a slight deviation at the initial instants. The slight deviation is probably

![](<liang2018_images/imageFile7.png>)

FIG. 7. Snapshots of droplet impact on a thin liquid ﬁlm with Re = 20 , We = 8000, and ρ l /ρ g = 1000. The time instants t ∗ have been normalized by the characteristic time D/U .

![](<liang2018_images/imageFile8.png>)

Re=500

Re=100

0

1/2

10 0

r/D=1.35(Ut/D) 1/2

r/D

-1

10

-2

-1

0

10

10

10 0

Ut/D

FIG. 8. The numerically predicted spreading radius versus the dimensionless time. The solid line represents the theoretical power law.

# IV. SUMMARY

Numerical modeling of two-phase flows with large density ratios is still a challenging task in the framework of the LB approach. In this paper, we propose a simple and accurate LB model for two-phase systems, which is capable of simulating large-density-ratio flows. The proposed LB model is based on the conservative phase-field equation, which involves a lowerorder diffusion term compared with the commonly used CahnHilliard equation in interface capturing. Therefore, the present model is expected to achieve a better numerical accuracy and stability. In addition, a force distribution function is also elaborately designed in this model such that it contains only onenonlocalmacroscopicquantity,whichismuchsimplerthan the previous phase-field-based LB models [10-12,25,27,28]. The multiscale analysis also demonstrates that both the conservative Allen-Cahn equation and the incompressible NavierStokes equations can be derived correctly from the present model. To validate the present model, we first simulated two basic steady problems of static droplet and layered Poiseuille flows, which have their own analytical solutions. In the former test, it is found that the present model can accurately capture the density field distributions in the bulk regions and also across the interface at the density ratio of 1000. In addition, it is also shown that the present model can obtain relatively small spurious velocities in the LB community, with the maximum magnitude of the order of 10 -9 . In the latter test, we simulated the channel flow with density ratios ranging from 10 to 1000, and also conducted detailed comparisons with the previous Allen-Cahn based LB models [27,28]. It is found that the present model can obtain satisfactory results in the velocity predictions, and is also more accurate than the previous LB models [27,28]. Next, we consider two dynamic problems of spinodal decomposition and droplet impact on a thin liquid film with a large density ratio of 1000. The phase separation process can be clearly observed in the system, which is in line with the expectation. The present model also successfully reproduces the classical splashing phenomenon, and the predicted spreading radius is found to exhibit the power law reported in the literature, which provides a good validation of the present LB model in dealing with complex high-density-ratio two-phase flows. Finally, we anticipate that our numerical method will be useful to scientific applications, such as microfluidics, material science, and oil recovery industry.

# ACKNOWLEDGMENTS

H.L. gratefully acknowledges insightful discussions with Professor Qing Li in the study of the droplet impact problem. This work is ﬁnancially supported by the National Natural Science Foundation of China under Grants No. 11602075, No. 51576079, and No. 51776068.

# APPENDIX: CHAPMAN-ENSKOG ANALYSIS OF THE PRESENT MODEL

TheChapman-Enskoganalysisisnowperformedtodemonstrate the consistency of the LB evolution equation ( 15 ) with the incompressible Navier-Stokes equations. The moment conditions are ﬁrst given based on the expressions of the equilibrium and force distribution functions:

$$
\text { the incomplete " Naveller-Stokes" equations. } & \text { the moment } \\ \text { conditions are first given based on the expressions of the } & \text {equivibility and force distribution functions: } \\ & \sum _ { i } g _ { i } ^ { e q } = 0 , \ \sum _ { i } c _ { i \alpha } g _ { i } ^ { e q } = \rho u _ { \alpha } , & \text {The re } \\ & \sum _ { i } c _ { i \alpha } c _ { i \beta } g _ { i } ^ { e q } = \rho u _ { \alpha } u _ { \beta } + p \delta _ { \alpha \beta } , & \\ & \sum _ { i } c _ { i \alpha } c _ { i \beta } c _ { i \gamma } g _ { i } ^ { e q } = \rho c _ { s } ^ { 2 } \Delta _ { \alpha \beta \gamma \theta } u _ { \theta } , & ( A 1 ) \\ & \sum _ { i } G _ { i } = \left ( 1 - \frac { 1 } { 2 \tau _ { g } } \right ) u _ { \alpha } \partial _ { \alpha } \rho , & \text {from} \\ & \sum _ { i } c _ { i \alpha } G _ { i } = \left ( 1 - \frac { 1 } { 2 \tau _ { g } } \right ) F _ { \alpha } , & \text { (A2)} \\ & \Lambda = \colon \sum _ { i } c _ { i \alpha } c _ { i \beta } G _ { i } = \left ( 1 - \frac { 1 } { 2 \tau _ { g } } \right ) [ u _ { \alpha } \partial _ { \beta } ( \rho c _ { s } ^ { 2 } ) \\ & \quad + u _ { \beta } \partial _ { \alpha } ( \rho c _ { s } ^ { 2 } ) + ( u _ { \gamma } \partial _ { \gamma } \rho c _ { s } ^ { 2 } ) \delta _ { \alpha \beta } ] , & ( A 2 ) \\ & \quad ( A 2 ) \\ & \quad \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \
$$

          where δ αβ is the Kronecker delta function,   αβγθ = δ αβ δ γθ + δ αγ δ βθ + δ αθ δ βγ . To derive the macroscopic equations, we expand the particle distribution function, the time and space derivatives, and the force in consecutive scales of   ,

$$
g _ { i } & = g _ { i } ^ { ( 0 ) } + \epsilon g _ { i } ^ { ( 1 ) } + \epsilon ^ { 2 } g _ { i } ^ { ( 2 ) } + \cdots , \quad ( A 3 a ) \\ & \\ & 0 \quad , \quad 0 \quad , \quad 2 0 \quad , \quad 0 \quad , \quad 0 \quad , \quad ( A 2 1 )
$$

$$
\partial _ { t } & = \epsilon \partial _ { t _ { 1 } } + \epsilon ^ { 2 } \partial _ { t _ { 2 } } , \ \partial _ { \alpha } = \epsilon \partial _ { 1 \alpha } , & & ( A 3 b ) \\ F & = \epsilon F ( 1 ) & & ( A 3 o )
$$

$$
F _ { \alpha } = \epsilon F _ { \alpha } ^ { ( 1 ) } , & & ( A 3 c ) \\
$$

where   is a small expansion parameter. Applying the Taylor expansion to Eq. ( 15 ), and substituting Eq. ( A3 ) into the expanded result, we can obtain the following multiscale equations:

$$
\epsilon ^ { 0 } \colon g _ { i } ^ { ( 0 ) } = g _ { i } ^ { ( q ) } , & & ( A 4 a ) & & \text {where} \\
$$

$$
\epsilon ^ { 1 } \colon D _ { 1 i } g _ { i } ^ { ( 0 ) } = - \frac { 1 } { \tau _ { g } \delta _ { i } } g _ { i } ^ { ( 1 ) } + G _ { i } ^ { ( 1 ) } , \quad \ \ ( A 4 b ) \quad \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \
$$

$$
\epsilon ^ { 2 } \colon \partial _ { t _ { 2 } } g _ { i } ^ { ( 0 ) } + D _ { 1 i } g _ { i } ^ { ( 1 ) } + \frac { \delta _ { t } } { 2 } D _ { 1 i } ^ { 2 } g _ { i } ^ { ( 0 ) } = - \frac { 1 } { \tau _ { g } \delta _ { t } } g _ { i } ^ { ( 2 ) } , \quad ( A 4 c )
$$

where D 1 i = ∂ t 1 + c iα ∂ 1 α . The substitution of Eq. ( A4b ) into Eq. ( A4c ) yields

$$
\partial _ { t _ { 2 } } g _ { i } ^ { ( 0 ) } + D _ { 1 i } \left ( 1 - \frac { 1 } { 2 \tau _ { g } } \right ) g _ { i } ^ { ( 1 ) } = - \frac { 1 } { \tau _ { g } \delta _ { t } } g _ { i } ^ { ( 2 ) } - \frac { \delta _ { t } } { 2 } D _ { 1 i } G _ { i } ^ { ( 1 ) } . \ ( A 5 ) \quad \text {with} \quad \begin{matrix} \text {which} \\ \end{matrix}
$$

Following Refs. [ 12 , 38 ], the zero-order moment of g i can be deﬁned as

$$
\sum _ { k } g _ { k } = - \frac { \delta _ { t } } { 2 } u _ { \alpha } \partial _ { \alpha } \rho . \\ \expansion form { \tilde { u } } ( A 3 ) \, \text {to} \, \text {Fqs} \, ( 2 1 a ) \, \text {and} \, ( A 6 )
$$

Applying the expansion formula ( A3 ) to Eqs. ( 21a ) and ( A6 ), one can easily derive

$$
1 \text { one easily derive} \\ \sum _ { i } g _ { i } ^ { ( 1 ) } = - \frac { \delta _ { t } } { 2 } u _ { \alpha } \partial _ { 1 \alpha } \rho , \ \sum _ { i } g _ { i } ^ { ( n ) } = 0 \ \ ( n \geqslant 2 ) , \quad ( A 7 ) \\ \sum _ { i } c _ { i \alpha } g _ { i } ^ { ( 1 ) } = - \frac { \delta _ { t } } { 2 } F _ { \alpha } ^ { ( 1 ) } , \\ \sum _ { i } c _ { i \alpha } g _ { i } ^ { ( n ) } = 0 \ \ ( n \geqslant 2 ) . \\ \intertext { t h e r e c o v e r e d e q u a t i o n s at e c l a s c a n b e o t a i n e d by s u m m i n g } \text {The recovered equations at } \epsilon \text { scale can be obtained by summing} \\ \text {Eq. (A4b) and Eq. (A4b) } \times c _ { i } \text { over } i \text { respectively}
$$

Therecoveredequationsat   scalecanbeobtainedbysumming Eq. ( A4b ) and Eq. ( A4b ) × c iβ over i , respectively,

$$
\partial _ { 1 \alpha } u _ { \alpha } = 0 ,
$$

$$
\partial _ { t _ { 1 } } ( \rho u _ { \beta } ) + \partial _ { 1 \alpha } ( \rho u _ { \alpha } u _ { \beta } + p \delta _ { \alpha \beta } ) = F _ { \beta } ^ { ( 1 ) } .
$$

Similarly, the recovered equations at   2 scale can be derived from Eq. ( A5 ):

$$
\partial _ { t _ { 1 } } \left ( - \frac { \delta _ { t } } { 2 } u _ { \alpha } \partial _ { 1 \alpha } \rho \right ) + \partial _ { 1 \alpha } \left ( - \frac { \delta _ { t } } { 2 } F _ { \alpha } ^ { ( 1 ) } \right ) \\ = - \frac { \delta _ { t } } { 2 } \left [ \partial _ { t _ { 1 } } ( u _ { \alpha } \partial _ { 1 \alpha } \rho ) + \partial _ { 1 \alpha } F _ { \alpha } ^ { ( 1 ) } \right ] \quad ( A 1 1 ) \\ \partial _ { t _ { 2 } } ( \rho u _ { \beta } ) + \left ( 1 - \frac { 1 } { 2 } \right ) \partial _ { t _ { 1 } \alpha } \Pi ^ { ( 1 ) } = - \frac { \delta _ { t } } { 2 } \partial _ { t _ { 2 } } \Lambda ^ { ( 1 ) } , \quad ( A 1 2 )
$$

$$
\partial _ { t _ { 2 } } ( \rho u _ { \beta } ) + \left ( 1 - \frac { 1 } { 2 \tau _ { g } } \right ) \partial _ { 1 \alpha } \Pi ^ { ( 1 ) } = - \frac { \delta _ { t } } { 2 } \partial _ { 1 \alpha } \Lambda ^ { ( 1 ) } , \quad ( A 1 2 ) \\
$$

where   (1) =   i c iα c iβ g (1) i is the ﬁrst-order momentum ﬂux tensor determined below, and   =    (1) . From Eq. ( A4b ), one can get

$$
\Pi ^ { ( 1 ) } & = \sum _ { i } c _ { i \alpha } c _ { i \beta } g _ { i } ^ { ( 1 ) } = - \tau _ { g } \delta _ { t } \sum _ { i } c _ { i \alpha } c _ { i \beta } [ D _ { 1 i } g _ { i } ^ { ( 0 ) } - G _ { i } ^ { ( 1 ) } ] \\ & = - \tau _ { g } \delta _ { t } c _ { s } ^ { 2 } [ \partial _ { 1 \alpha } ( \rho u _ { \beta } ) + \partial _ { 1 \beta } ( \rho u _ { \alpha } ) + ( \partial _ { 1 \gamma } \rho u _ { \gamma } ) \delta _ { \alpha \beta } ] \\ & \quad + \tau _ { g } \delta _ { t } \Lambda ^ { ( 1 ) } , \\ \text {where the terms of } O ( \delta _ { t } M a ^ { 2 } ) \text { have been neglected under the }
$$

where the terms of O ( δ t Ma 2 ) have been neglected under the incompressible limit. Substituting Eq. ( A13 ) into Eq. ( A12 ), one can simplify Eq. ( A12 ) as

$$
\partial _ { t _ { 2 } } ( \rho u _ { \beta } ) - \partial _ { 1 \alpha } [ v \rho ( \partial _ { 1 \alpha } u _ { \beta } + \partial _ { 1 \beta } u _ { \alpha } ) ] = 0 , \quad ( A 1 4 ) \\
$$

where ν = c 2 s δ t ( τ g − 1 2 ) is the kinematic viscosity. Combining Eqs.( A9 )and( A11 )at   and   2 scales,togetherwithEqs.( A10 ) and ( A14 ), we have

$$
\partial _ { \alpha } u _ { \alpha } = 0 , \quad ( A 1 5 )
$$

$$
\partial _ { t } ( \rho u _ { \beta } ) + \partial _ { \alpha } ( \rho u _ { \alpha } u _ { \beta } + p \delta _ { \alpha \beta } ) \\ = \partial _ { 1 \alpha } [ v \rho ( \partial _ { 1 \alpha } u _ { \beta } + \partial _ { 1 \beta } u _ { \alpha } ) ] + F _ { \beta } , \quad ( A 1 6 ) \\ \intertext { o n c l e r y s h o w s } \intertext { a n d } \intertext { o n t h e i n c h r p a r s i b l e N a v i o n } \intertext { s u p t h e r } \intertext { o n t h e i n c h r p a r s i b l e N a v i o n } \intertext { s u p t h e r } \intertext { o n t h e i n c h r p a r s i b l e N a v i o n } \intertext { s u p t h e r } \intertext { o n t h e i n c h r p a r s i b l e N a v i o n } \intertext { s u p t h e r } \intertext { o n t h e i n c h r p a r s i b l e N a v i o n } \intertext { s u p t h e r } \intertext { o n t h e i n c h r p a r s i b l e N a v i o n } \intertext { s u p t h e r } \intertext { o n t h e i n c h r p a r s i b l e N a v i o n } \intertext { s u p t h e r } \intertext { o n t h e i n c h r p a r s i b l e N a v i o n } \intertext { s u p t h e r } \intertext { o n t h e i n c h r p a r s i b l e N a v i o n } \intertext { s u p t h e r } \intertext { o n t h e i n c h r p a r s i b l e N a v i o n } \intertext { s u p t h e r } \intertext { o n t h e i n c h r p a r s i b l e N a v i o n } \intertext { s u p t h e r } \intertext { o n t h e i n c h r p a r s i b l e N a v i o n } \intertext { s u p t h e r } \intertext { o n t h e i n c h r p a r s i b l e N a v i o n } \intertext { s u p t h e r } \intertext { o n t h e i n c h r p a r s i b l e N a v i o n } \intertext { s u p t h e r } \intertext { o n t h e i n c h r p a r s i b l e N a v i o n } \intertext { s u p t h e r } \intertext { o n t h e i n c h r p a r s i b l e N a v i o n } \intertext { s u p t h e r } \intertext { o n t h e i n c h r p a r s i b l e N a v i o n } \intertext { s u p t h e r } \intertext { o n t h e i n c h r p a r s i b l e N a v i o n } \intertext { s u p t h e r } \intertext { o n t h e i n c h r p a r s i b l e N a v i o n } \intertext { s u p t h e r } \intertext { o n t h e i n c h r p a r s i b l e N a v i o n } \intertext { s u p t h e r } \intertext { o n t h e i n c h r p a r s i b l e N a v i o n } \intertext { s u p t h e r } \intertext { o n t h e i n c h r p a r s i b l e N a v i o n } \intertext { s u p t h e r } \intertext { o n t h e i n c h r p a r s i b l e N a v i o n } \intertext { s u p t h e r } \intertext { o n t h e i n c h r p a r s i b l e N a v i o n } \intertext { s u p t h e r } \intertext { o n t h e i n c h r p a r s i b l e N a v i o n } \intertext { s u p t h e r } \intertext { o n t h e i n c h r p a r s i b l e N a v i o n } \intertext { s u p t h e r } \intertext { o n t h e i n c h r p a r s i b l e N a v i o n } \intertext { s u p t h e r } \intertext { o n t h e i n c h r p a r s i b l e N a v i o n } \intertext { s u p t h e r } \intertext { o n t h e i n c h r p a r s i b l e N a v i o n } \intertext { s u p t h e r } \intertext { o n t h e i n c h r p a r s i b l e N a v i o n } \intertext { s u p t h e r } \intertext { o n t h e i n c h r p a r s i b l e N a v i o n } \intertext { s u p t h e r } \intertext { o n t h e i n c h r p a r s i b l e N a v i o n } \intertext { s u p t h e r } \intertext { o n t h e i n c h r p a r s i b l e N a v i o n } \intertext { s u p t h e r } \intertext { o n t h e i n c h r p a r s i b l e N a v i o n } \intertext { s u p t h e r } \intertext { o n t h e i n c h r p a r s i b l e N a v i o n } \intertext { s u p t h e r } \intertext { o n t h e i n c h r p a r s i b l e N a v i o n } \intertext { s u p t h e r } \intertext { o n t h e i n c h r p a r s i b l e N a v i o n } \intertext { s u p t h e r } \intertext { o n t h e i n c h r p a r s i b l e N a v i o n } \intertext { s u p t h e r } \intertext { o n t h e i n c h r p a r s i b l e N a v i o n } \intertext { s u p t h e r } \intertext { o n t h e i n c h r p a r s i b l e N a v i o n } \intertext { s u p t h e r } \intertext { o n t h e i n c h r p a r s i b l e N a v i o n } \intertext { s u p t h e r } \intertext { o n t h e i n c h r p a r s i b l e N a v i o n } \intertext { s u p t h e r } \intertext { o n t h e i n c h r p a r s i b l e N a v i o n } \intertext { s u p t h e r } \intertext { o n t h e i n c h r p a r s i b l e N a v i o n } \intertext { s u p t h e r } \intertext { o n t h e i n c h r p a r s i b l e N a v i o n } \intertext { s u p t h e r } \intertext { o n t h e i n c h r p a r s i b l e N a v i o n } \intertext { s u p t h e r } \intertext { o n t
$$

which clearly shows that the incompressible Navier-Stokes equations can be exactly recovered from the present LB model.

[1] S. Succi, The Lattice Boltzmann Equation for Fluid Dynamics and Beyond (Oxford University Press, Oxford, 2001).

[2] Z. L. Guo and C. Shu, Lattice Boltzmann Method and its Applications in Engineering (World Scientiﬁc, Singapore, 2013).

[3] T. Krüger, H. Kusumaatmaja, A. Kuzmin, O. Shardt, G. Silva, and E. M. Viggen, The Lattice Boltzmann Method: Principles and Practice (Springer, Berlin, 2016).

- [4] B. C. Shi and Z. L. Guo, Lattice Boltzmann model for nonlinear convection-diffusion equations, Phys. Rev. E 79 , 016701 ( 2009 ).
- [5] Z. H. Chai, B. C. Shi, and Z. L. Guo, A multiple-relaxationtime lattice Boltzmann model for general nonlinear anisotropic convection-diffusion equations, J. Sci. Comput. 69 , 355 ( 2016 ).


[6] A. K. Gunstensen, D. H. Rothman, S. Zaleski, and G. Zanetti, Lattice Boltzmann model of immiscible ﬂuids, Phys. Rev. A 43 , 4320 ( 1991 ).

[7] X. Shan and H. Chen, Lattice Boltzmann model for simulating ﬂows with multiple phases and components, Phys. Rev. E 47 , 1815 ( 1993 ).

[8] M. Swift, W. Osborn, and J. Yeomans, Lattice Boltzmann Simulation of Nonideal Fluids, Phys. Rev. Lett. 75 , 830 ( 1995 ).

[9] X. He, S. Chen, and R. Zhang, A lattice Boltzmann scheme for incompressible multiphase ﬂow and its application in simulation ofRayleigh-Taylorinstability, J.Comput.Phys. 152 , 642 ( 1999 ).

[10] T. Lee and C. L. Lin, A stable discretization of the lattice Boltzmann equation for simulation of incompressible two-phase ﬂows at high density ratio, J. Comput. Phys. 206 , 16 ( 2005 ).

- [11] T.LeeandL.Liu,LatticeBoltzmannsimulationsofmicron-scale drop impact on dry surfaces, J. Comput. Phys. 229 , 8045 ( 2010 ).
- [12] H. Liang, B. C. Shi, Z. L. Guo, and Z. H. Chai, Phase-ﬁeld-based multiple-relaxation-time lattice Boltzmann model for incompressible multiphase ﬂows, Phys. Rev. E 89 , 053320 ( 2014 ).


[13] H. Liang, Z. H. Chai, B. C. Shi, Z. L. Guo, and T. Zhang, Phase-ﬁeld-based lattice Boltzmann model for axisymmetric multiphase ﬂows, Phys. Rev. E 90 , 063311 ( 2014 ).

[14] H. Liu, Q. J. Kang, C. R. Leonardi, S. Schmieschek, A. Narváez, B. D. Jones, J. R. Williams, A. J. Valocchi, and J. Harting, Multiphase lattice Boltzmann simulations for porous media applications, Comput. Geosci. 20 , 777 ( 2016 ).

[15] Q. Li, H. K. Luo, Q. J. Kang, Y. L. He, Q. Chen, and Q. Liu, Lattice Boltzmann methods for multiphase ﬂow and phasechange heat transfer, Prog. Energy Combust. Sci. 52 , 62 ( 2016 ).

[16] T. Inamuro, T. Ogata, S. Tajima, and N. Konishi, A lattice Boltzmann method for incompressible two-phase ﬂows with large density differences, J. Comput. Phys. 198 , 628 ( 2004 ).

[17] J. A. Sethian, Level Set Methods and Fast Marching Methods: Evolving Interfaces in Computational Geometry, Fluid Mechanics, Computer Vision, and Materials Science (Cambridge University Press, Cambridge, UK, 1999).

[18] P.YuanandL.Schaefer,EquationsofstateinalatticeBoltzmann model, Phys. Fluids 18 , 042101 ( 2006 ).

[19] Q. Li, K. H. Luo, and X. J. Li, Lattice Boltzmann modeling of multiphase ﬂows at large density ratio with an improved pseudopotential model, Phys. Rev. E 87 , 053301 ( 2013 ).

[20] Y. Ba, H. Liu, Q. Li, Q. Kang, and J. Sun, Multiple-relaxationtime color-gradient lattice Boltzmann model for simulating twophase ﬂows with high density ratio, Phys. Rev. E 94 , 023310 ( 2016 ).

[21] D. Jacqmin, Calculation of two-phase Navier-Stokes ﬂows using phase-ﬁeld modeling, J. Comput. Phys. 155 , 96 ( 1999 ).

[22] H.W.Zheng,C.Shu,andY.T.Chew,AlatticeBoltzmannmodel for multiphase ﬂows with large density ratio, J. Comput. Phys. 218 , 353 ( 2006 ).

[23] A. Fakhari and M. H. Rahimian, Phase-ﬁeld modeling by the method of lattice Boltzmann equations, Phys. Rev. E 81 , 036707 ( 2010 ).

[24] Q. Lou, Z. L. Guo, and B. C. Shi, Effects of force discretization on mass conservation in lattice Boltzmann equation for twophase ﬂows, Europhys. Lett. 99 , 64005 ( 2012 ).

[25] Y. Q. Zu and S. He, Phase-ﬁeld-based lattice Boltzmann model for incompressible binary ﬂuid systems with density and viscosity contrasts, Phys. Rev. E 87 , 043301 ( 2013 ).

[26] Y. Wang, C. Shu, H. B. Huang, and C. T. Teo, Multiphase lattice Boltzmann ﬂux solver for incompressible multiphase ﬂows with large density ratio, J. Comput. Phys. 280 , 404 ( 2015 ).

[27] F. Ren, B. W. Song, M. C. Sukop, and H. B. Hu, Improved lattice Boltzmann modeling of binary ﬂow based on the conservative Allen-Cahn equation, Phys. Rev. E 94 , 023311 ( 2016 ).

[28] A. Fakhari and D. Bolster, Diffuse interface modeling of three-phase contact line dynamics on curved boundaries: A lattice Boltzmann model for large density and viscosity ratios, J. Comput. Phys. 334 , 620 ( 2017 ).

[29] M.Geier,A.Fakhari,andT.Lee,Conservativephase-ﬁeldlattice Boltzmann model for interface tracking equation, Phys. Rev. E 91 , 063309 ( 2015 ).

[30] H. L. Wang, Z. H. Chai, B. C. Shi, and H. Liang, Comparative study of the lattice Boltzmann models for Allen-Cahn and CahnHilliard equations, Phys. Rev. E 94 , 063304 ( 2016 ).

- [31] Y. Sun and C. Beckermann, Sharp interface tracking using the phase-ﬁeld equation, J. Comput. Phys. 220 , 626 ( 2007 ).
- [32] P. H. Chiu and Y. T. Lin, A conservative phase ﬁeld method for solving incompressible two-phase ﬂows, J. Comput. Phys. 230 , 185 ( 2011 ).


[33] Y. H. Qian, D. d’Humires, and P. Lallemand, Lattice BGK models for Navier-Stokes equation, Europhys. Lett. 17 , 479 ( 1992 ).

[34] I. Ginzburg, F. Verhaeghe, and D. d’Humières, Two-relaxationtimelatticeBoltzmannscheme:Aboutparametrization,velocity, pressure and mixed boundary conditions, Commun. Comput. Phys. 3 , 427 (2008).

[35] P. Lallemand and L. S. Luo, Theory of the lattice Boltzmann method: Dispersion, dissipation, isotropy, Galilean invariance, and stability, Phys. Rev. E 61 , 6546 ( 2000 ).

[36] S. O. Unverdi and G. Tryggvason, A front-tracking method for viscous,incompressible,multi-ﬂuidﬂows, J.Comput.Phys. 100 , 25 ( 1992 ).

[37] J. Kim, A continuous surface tension force formulation for diffuse-interface models, J. Comput. Phys. 204 , 784 ( 2005 ).

[38] H. Liang, B. C. Shi, and Z. H. Chai, Lattice Boltzmann modeling of three-phase incompressible ﬂows, Phys. Rev. E 93 , 013308 ( 2016 ).

[39] Y. K. Wei, Z. D. Wang, H. S. Dou, and Y. H. Qian, A novel twodimensional coupled lattice Boltzmann model for incompressibleﬂowinapplicationofturbulenceRayleigh-Taylorinstability, Comput. Fluids 156 , 97 ( 2017 ).

[40] Z. L. Guo, C. G. Zheng, and B. C. Shi, Discrete lattice effects on the forcing term in the lattice Boltzmann method, Phys. Rev. E 65 , 046308 ( 2002 ).

[41] Y. K. Wei, Z. D. Wang, J. F. Yang, H. S. Dou, and Y. H. Qian, A simple lattice Boltzmann model for turbulence Rayleigh-Bénard thermal convection, Comput. Fluids 118 , 167 ( 2015 ).

[42] D. M. Anderson, G. B. McFadden, and A. A. Wheeler, Diffuseinterface methods in ﬂuid mechanics, Annu. Rev. Fluid Mech. 30 , 139 ( 1998 ).

[43] Z. H. Chai and T. S. Zhao, Effect of the forcing term in the multiple-relaxation-time lattice Boltzmann equation on the shear stress or the strain rate tensor, Phys. Rev. E 86 , 016705 ( 2012 ).

[44] W. A. Yong and L. S. Luo, Accuracy of the viscous stress in the lattice Boltzmann equation with simple boundary conditions, Phys. Rev. E 86 , 065701(R) ( 2012 ).

[45] Z. L. Guo, C. G. Zheng, and B. C. Shi, Force imbalance in lattice Boltzmann equation for two-phase ﬂows, Phys. Rev. E 83 , 036707 ( 2011 ).

[46] Z. Yu and L. S. Fan, Multirelaxation-time interaction-potentialbased lattice Boltzmann model for two-phase ﬂow, Phys. Rev. E 82 , 046708 ( 2010 ).

[47] H. B. Huang and X. Y. Lu, Relative permeabilities and coupling effects in steady-state gas-liquid ﬂow in porous media: A lattice Boltzmann study, Phys. Fluids 21 , 092104 ( 2009 ).

[48] H. Liang, B. C. Shi, and Z. H. Chai, An efﬁcient phaseﬁeld-based multiple-relaxation-time lattice Boltzmann model for three-dimensional multiphase ﬂows, Comput. Math. Appl. 73 , 1524 ( 2017 ).

[49] J. W. Cahn, Phase separation by spinodal decomposition in isotropic systems, J. Chem. Phys. 42 , 93 ( 1965 ).

[50] Y. B. Gan, A. G. Xu, G. C. Zhang, and S. Succi, Discrete Boltzmann modeling of multiphase ﬂows: Hydrodynamic and thermodynamic non-equilibrium effects, Soft Matter 11 , 5336 ( 2015 ).

- [51] A. L. Yarin, Drop impact dynamics: Splashing, spreading, receding, bouncing..., Annu. Rev. Fluid Mech. 38 , 159 ( 2006 ).
- [52] G. Coppola, G. Rocco, and L. Luca, Insights on the impact of a plane drop on a thin liquid ﬁlm, Phys. Fluids 23 , 022105 ( 2011 ).


[53] J. S. Lee, B. M. Weon, J. H. Je, and K. Fezzaa, How Does an Air Film Evolve into a Bubble During Drop Impact?, Phys. Rev. Lett. 109 , 204501 ( 2012 ).

[54] M. J. Thoraval, K. Takehara, T. G. Etoh, and S. T. Thoroddsen, Drop impact entrapment of bubble rings, J. Fluid Mech. 724 , 234 ( 2013 ).

[55] C. Josseranda and S. Zaleskib, Droplet splashing on a thin liquid ﬁlm, Phys. Fluids 15 , 1650 ( 2003 ).

