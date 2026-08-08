![](<Yang 等 - 2024 - Three-dimensional pore-scale study of methane hydrate dissociation mechanisms based on micro-CT imag_images/imageFile1.png>)

# Three-dimensional pore-scale study of methane hydrate dissociation mechanisms based on micro-CT images

###### Junyu Yang,1 Qianghui Xu,2,* Xuan Kou,3 Geng Wang,1 Timan Lei,1 Yi Wang,3 Xiaosen Li,3 and Kai H. Luo1,4,*

*Correspondence: xuqh12@bit.edu.cn (Q.X.); k.luo@ucl.ac.uk (K.L.) Received: January 27, 2024; Accepted: February 19, 2024; Published Online: February 23, 2024; https://doi.org/10.59717/j.xinn-energy.2024.100015 © 2024 The Author(s). This is an open access article under the CC BY-NC-ND license (http://creativecommons.org/licenses/by-nc-nd/4.0/).

###### ARTICLE

GRAPHICAL ABSTRACT

![](<Yang 等 - 2024 - Three-dimensional pore-scale study of methane hydrate dissociation mechanisms based on micro-CT imag_images/imageFile2.png>)

PUBLIC SUMMARY

- ■ Methane hydrate dissociation mechanisms within actual sediment pore structure are revealed.
- ■ The hydrate pore habits influence the dissociation mechanisms and upscaling models.
- ■ A pore-scale numerical framework for methane hydrate dissociation is established.
- ■ A good agreement between pore-scale experiments and numerical simulations is achieved.
- ■ Upscaling parameters for the methane hydrate production forecast are provided.


energy

![](<Yang 等 - 2024 - Three-dimensional pore-scale study of methane hydrate dissociation mechanisms based on micro-CT imag_images/imageFile3.png>)

###### ARTICLE

# Three-dimensional pore-scale study of methane hydrate dissociation mechanisms based on micro-CT images

###### Junyu Yang,1 Qianghui Xu,2,* Xuan Kou,3 Geng Wang,1 Timan Lei,1 Yi Wang,3 Xiaosen Li,3 and Kai H. Luo1,4,*

1Department of Mechanical Engineering, University College London, Torrington Place, London WC1E 7JE, UK 2School of Mechanical Engineering, Beijing Institute of Technology, Beijing 100081, China 3Guangzhou Institute of Energy Conversion, Chinese Academy of Sciences, Guangzhou 510640, China 4Center for Combustion Energy, Tsinghua University, Beijing 100084, China

*Correspondence: xuqh12@bit.edu.cn (Q.X.); k.luo@ucl.ac.uk (K.L.) Received: January 27, 2024; Accepted: February 19, 2024; Published Online: February 23, 2024; https://doi.org/10.59717/j.xinn-energy.2024.100015 © 2024 The Author(s). This is an open access article under the CC BY-NC-ND license (http://creativecommons.org/licenses/by-nc-nd/4.0/). Citation: Yang J., Xu Q., Kou X., et al., (2024). Three-dimensional pore-scale study of methane hydrate dissociation mechanisms based on micro-CT images. The Innovation Energy 1(1): 100015.

Methane hydrate is a promising source of alternative energy. An in-depth understanding of the hydrate dissociation mechanism is crucial for the efficient extraction. In the present work, a comprehensive set of pore-scale numerical studies of hydrate dissociation mechanisms is presented. Porescale lattice Boltzmann (LB) models are proposed to simulate the multiphysics process during methane hydrate dissociation. The numerical simulations employ the actual hydrate sediment pore structure obtained by the micro-CT imaging. Experimental results of xenon hydrate dissociation are compared with the numerical simulations, indicating that the observed hydrate pore habits evolution is accurately captured by the proposed LB models. Furthermore, simulations of methane hydrate dissociation under different sediment water saturations, fluid flow rates and thermal conditions are conducted. Heat and mass transfer limitations both have significant effects on the methane hydrate dissociation rate. The bubble movement can further influence the dissociation process. Dissociation patterns can be divided into three categories, uniform, non-uniform and wormholing. The fluid flow impacts hydrate dissociation rates differently in threedimensional real structures compared to two-dimensional idealized ones, influenced by variations in hydrate pore habits and flow properties. Finally, upscaling investigations are conducted to provide the permeability and kinetic models for the representative elementary volume (REV)-scale production forecast. Due to the difference in the hydrate pore habits and dissociation mechanisms, the three-dimensional upscaling results contrast with prior findings from two-dimensional studies. The present work provides a paradigm for pore-scale numerical simulation studies on the hydrate dissociation, which can offer theoretical guidance on efficient hydrate extraction.

###### INTRODUCTION

In addressing the challenges of global warming and energy shortages,1-3 methane hydrate4 has emerged as a promising alternative energy resource.5,6 This crystalline compound, comprising methane molecules within hydrogenbonded water lattices,4 is abundant in deep marine sediments7-9 and terrestrial permafrost regions.10 Its vast reserves, estimated to be double that of other fossil fuels combined,11,12 have attracted global attention for its potential in clean energy production.13-16 However, extracting methane hydrate is complex, involving methods like thermal stimulation,17 depressurisation,18 and inhibitor injection,19 all of which aim to disrupt the thermodynamic equilibrium necessary for methane hydrate stability. The efficiency of these extraction methods remains limited20 due to the intricate dissociation mechanisms and variable reservoir conditions. The dissociation of methane hydrate is a complex, multi-scale and multiphysical process.21 In the sediment with diverse hydrate pore habits, methane hydrate dissociation reaction occurs accompanied by multiphase gas-water flows, interfacial heat and mass transfer and structural changes. These processes influence the thermodynamic conditions within the sediment, further complicating the recovery process.22 Therefore, understanding these multiphysical processes is vital to enhance extraction techniques. Research into methane hydrate dissociation spans multiple scales,13,23-25 with pore-scale26,27 studies offering detailed insights into the sediment porous media. Techniques like microfluidic observations28-31 and micro-CT imaging32-36 have been instrumental in visualizing

the dynamic evolution of hydrate pore habits. These studies have revealed consistent patterns in hydrate dissociation, such as the gradual shrinkage of hydrate clusters and the transition of surface shapes, influenced by mass and heat transfer limitations.33,37-39 However, the limitations of these experimental approaches, particularly the constraints of micro-CT's spatial and temporal resolution and the use of xenon for enhanced image contrast, have raised questions about the extrapolation of these findings to methane hydrate reservoirs. This necessitates pore-scale numerical simulations for a comprehensive understanding of these multiphysics mechanisms.

The advancement in computing hardware and computational methods has facilitated pore-scale numerical investigations. These studies have revealed significant insights into the impacts of heat and mass transfer limitations on methane hydrate dissociation.40 21,36,41-48 Yet, most of these investigations rely on simplified two-dimensional structures, which fail to accurately represent the complexity of three-dimensional hydrate pore habits in real reservoirs. As such, there is a growing need for three-dimensional pore-scale numerical investigations that can offer more realistic insights into methane hydrate dissociation. Pore-scale studies also play a crucial role in developing foundational parameters for simulations at the representative elementary volume (REV) scale, which is commonly used in methane hydrate production forecasts.49,50 Accurately capturing the complex multiphysics phenomena within the pore structure is critical for these models. While substantial progress has been made in understanding permeability36,44-48,51 and kinetic models for dissociation reactions,22,25,52-54 further exploration is required, especially in the context of real hydrate reservoir structures.

To overcome these challenges, this work focuses on developing efficient and accurate physical and numerical models for three-dimensional porescale simulations based on actual hydrate sediment micro-CT structures. Utilizing the lattice Boltzmann (LB) method, known for its computational efficiency and ease in simulating physical processes in porous media,55-57 we establish comprehensive models that capture the multiphysics characteristics of methane hydrate dissociation. The numerical investigations, in conjunction with experimental results, will help clarify the effects of multiphysics mechanisms on hydrate dissociation, deepen our understanding through pore-scale simulations under various reservoir conditions, and inform upscaling work for more accurate methane hydrate production models.

###### MATERIALS AND METHODS Governing equations and numerical models

The present study delves into the dissociation of methane hydrate induced by depressurisation, encompassing a multitude of multiphysical processes within the pore structure. This includes the dynamics of gas-water multiphase flow, the intricacies of multiphase heat and mass transfer, the kinetics of hydrate dissociation, and the evolution of hydrate structure. To effectively simulate these complex processes, lattice Boltzmann models have been developed and meticulously employed. The governing equations that describe these multiphysical processes, along with the corresponding numerical models, are comprehensively detailed in Note S1 and S5.

###### Comparison of numerical and experimental results

We rigorously compare the results of our numerical simulations with micro-

![](<Yang 等 - 2024 - Three-dimensional pore-scale study of methane hydrate dissociation mechanisms based on micro-CT imag_images/imageFile4.png>)

Figure 1. Upscaling analysis (A) Conceptual diagram for the upscaling procedure. (B) Permeability model with hydrate saturation during methane hydrate dissociation. (C) Effective reaction surface area under different water saturation conditions during methane hydrate dissociation. (D) Comparison of the REV-scale results predicted by the upscaled kinetic model and the pore-scale numerical results of the hydrate dissociation rate.

CT images obtained from xenon hydrate dissociation experiments, as conducted by Kou et al.,33 to validate the accuracy and reliability of our numerical models. This comparison is pivotal for elucidating the complex heat and mass transfer mechanisms that underpin the experimental observations of hydrate dissociation. The micro-CT images before and after hydrate dissociation are captured and digitised for the numerical simulation (Figure S3). This comparison serves as a cornerstone for substantiating the credibility of our numerical models and provides deeper insights into the intricacies of hydrate dissociation phenomena. Detailed descriptions of both the experimental and numerical methodologies, including the specific parameters and conditions employed, are extensively documented in Note S2.

###### Micro-CT image for mechanisms analysis

For our numerical simulations aimed at elucidating the mechanisms of methane hydrate dissociation, we have utilized micro-CT images of methane hydrate sediment. The detailed structure of this sediment, as depicted in Figure S4, was meticulously captured through micro-CT imaging techniques by Chen et al..58 In these simulations, we have intentionally varied water saturation levels, incorporating a stochastic distribution of gas and water within the sediment (refer to Figure S4), to investigate the influence of water content

in the sediment on the dissociation process. Detailed information regarding the numerical settings, including the specific parameters and conditions employed in these simulations, is thoroughly outlined in Note S3.

###### Upscaling conception

The upscaling work in this research involves leveraging pore-scale studies to develop computational models for REV-scale production forecasting. In the REV-scale model, production predictions are primarily derived from solving mass and energy equations. Critical to this computational process are the kinetic and permeability parameters, which are associated with the flux and source terms. These parameters must be accurately obtained based on detailed pore-scale studies, as they play a pivotal role in the fidelity and precision of the REV-scale predictions. The governing equations of the REV-scale model, along with the intricacies of the upscaling concepts schematised from Figure 1A, are elucidated in comprehensive detail in Note S4.

###### RESULTS AND DISCUSSION Effect of mass transfer on the dissociation process

A comparison of the numerical and experimental results of xenon hydrate dissociation is conducted to elucidate the multiphysical mechanisms, as

###### energy

![](<Yang 等 - 2024 - Three-dimensional pore-scale study of methane hydrate dissociation mechanisms based on micro-CT imag_images/imageFile5.png>)

Figure 2. Comparison of numerical and experimental results (A) Numerical results of xenon hydrate dissociation including the hydrate structure, multiphase distribution, xenon concentration in water phase, xenon concentration in gas phase and temperature. (B) Comparison of the experimental and numerical results of hydrate structure at Sh=0.03 during dissociation.

described in Note S2. Details of the governing equations S1-S8 and numerical implementations can be found in Note S1. The numerical results, showcased in Figure 2, effectively mirror the experimental findings, particularly in the evolution of hydrate structure during dissociation. Initially, hydrates are present in patchy blocks within the pore space, and upon depressurization-

induced dissociation, these blocks shrink, forming concave surfaces and evolving into grain-bridging forms (t=200 s). As dissociation progresses, these forms eventually disappear, leaving behind central hydrate blocks interconnected through a water envelope (t=400-600 s), a phenomenon termed ‘hydrate-bridge’.

![](<Yang 等 - 2024 - Three-dimensional pore-scale study of methane hydrate dissociation mechanisms based on micro-CT imag_images/imageFile6.png>)

Figure 3. Numerical results of isothermal methane hydrate dissociation (A) Numerical results with Sw=0.20, Pe~O(10−2), including the phase distribution and concentration in water Cw. (B) Numerical results of the phase distribution under the condition of Sw=0.20, Pe~O(100). (C) Histograms and heat maps of hydrate dissociation rate in each block at the cross section of x=1.4 mm. (D) The isothermal hydrate dissociation curves under different water saturations and fluid flow rates.

The observed hydrate morphology evolution is primarily attributable to the mass transfer limitation imposed by the water layer. There is a significant difference in the concentration distribution characteristics of xenon in the gas and water phases. In the gas phase, xenon concentration remains consistently low and homogeneous, facilitating rapid hydrate dissociation. Conversely, in the water phase, a steeper concentration gradient is observed, as shown in the enlarged view in Figure S10. Near the gas-water interface, xenon concentrations are lower, while near the hydrate surface, higher concentrations slow down the dissociation rate. This disparity in xenon concentration, attributed to the varying diffusivity of gas molecules in different phases, leads to what is termed ‘mass transfer limitation’. The hydrate surface evolution reflects the contours of the gas-water interface, with faster dissociation rates closer to this interface. Figure 2B compares the numerical and experimental results using the structure of the hydrate after dissociation with the saturation of Sh=0.03 as a criterion. The numerical predictions closely align with the experimental observations, showcasing a mere 1.6% relative error. This substantiates the capability of the model to accurately replicate the dynamical evolution of hydrate dissociation for in-depth mechanical analysis.

Following the validation of the numerical models through the comparison of experimental and numerical results, we proceeded to simulate methane hydrate dissociation under varying conditions to elucidate its mechanisms. The structure of methane hydrate sediment is obtained from micro-CT images by Chen et al. 58 and the numerical settings are introduced in Note S3. The isothermal depressurisation process is firstly simulated with different initial water saturations and fluid flow rates incorporated to investigate the mass transfer effect. Figure 3 shows the numerical results under a lower water saturation of Sw=0.20. Two typical fluid flow rates are discussed with the Péclet number (Pe=UL/Dw) of Pe~O(10−2) and Pe~O(100). The temporal evolution of the phase distribution and methane concentration in the water phase is illustrated. As shown in Figure 3A, when the water saturation of the sediments is low, methane hydrate dissociation occurs primarily in the gas phase due to depressurisation. In contrast, methane molecules in the water phase face diffusive barriers across the gas-water interface, leading to a pronounced concentration gradient like the observations in Figure 2. The

concentration away from the gas-water interface maintains a high level in the water, consequently imposing a mass transfer limitation on the dissociation rate. Therefore, the methane hydrate covered by water decomposes much slower than that covered by gas. In the late dissociation stage at t=30 s, the hydrate enclosed in the gas phase is substantially decomposed, while a significant portion of water-enclosed hydrate remains undecomposed due to the mass transfer limitation.

When the flow rate is low, the gas-liquid interface basically does not move as illustrated in Figure 3A. Due to the interfacial tension, many bubbles are trapped immobile in the pore structure. Conversely, at higher flow rates, the inertia overcomes the interfacial tension, prompting a shift in the gas-water interface position. This movement permits hydrates previously encapsulated by water to re-establish contact with the gas phase. Hydrates that initially exhibited slow dissociation due to mass transfer limitations in the aqueous layer can now decompose more expediently upon gas phase exposure, as marked by the yellow circle in Figure 3B. The fluid flow plays a role in facilitating hydrate dissociation.

To gain insight into the hydrate structure evolution during dissociation, we partitioned the computational domain into numerous small blocks (10×10×10) and counted the hydrate dissociation rate 1-Sh/Sh0 at t=30 s in each block. Figure 3C counts the dissociation rates for each block in the cross-section at x=1.4 mm in both low and high fluid flow rate cases. The heat map of the hydrate dissociation rate distribution is also developed. The histograms reveal that the hydrate dissociates comprehensively across different flow rates. In approximately half the blocks, the dissociation rate surpasses 90%. This observation is attributed to the sediment low water saturation, enabling most of the hydrate to contact with the gas phase. The effect of mass transfer limitation in water is not significant. The heat map indicates that rapid dissociation occurs in most blocks, with only a select few enveloped by water exhibiting a reduced dissociation rate. Thus, the hydrate structure evolution is relatively homogeneous in the reservoir. At high flow rates, the number of blocks with decomposition rates above 90% increased. The heat map shows that at higher flow rates, the dissociation rate of some blocks increases compared to the lower flow rate case. This is caused by the movement of the gas-liquid interface, which weakens the mass transfer limi-

###### energy

![](<Yang 等 - 2024 - Three-dimensional pore-scale study of methane hydrate dissociation mechanisms based on micro-CT imag_images/imageFile7.png>)

- Figure 4. Numerical results of non-isothermal methane hydrate dissociation (A) Numerical results with Sw=0.40, Pe~O(10−2), including the phase distribution and concentra-


tion in water Cw. (B) Histograms and heat maps of hydrate dissociation rate in each block at the cross section of x=1.4 mm. (C) The non-isothermal hydrate dissociation curves under different water saturations and fluid flow rates. (D) Temperature variation curves for non-isothermal hydrate dissociation.

tation in water. Overall, regardless of the flow rate, the full field in the sediments maintained a high dissociation rate at low water saturation conditions.

With an increase in sediment-water saturation, the dissociation process shows a different pattern. As shown in Figure S11(a) with higher water saturation Sw=0.60 and low fluid flow rate, most of the hydrates in the sediments are encapsulated by the water phase. The concentration map shows that methane molecules are difficult to transport into the gas phase due to the low diffusivity in water. Consequently, elevated methane concentrations persist at numerous locations within the water phase, which subsequently suppresses hydrate dissociation. From the phase distribution diagram in Figure S11(a), most of the hydrate in the sediment remains, and only a small portion of the hydrate that is in contact with the gas phase is decomposed. The prominence of mass transfer limitations becomes evident in influencing the hydrate dissociation rates. When the flow rate is low, the bubbles are trapped in the pore structure due to capillary forces, preventing any significant gaswater interface movement. After the hydrate covered by the gas phase is consumed, reservoir hydrate dissociation relies mainly on slow dissociation in the water phase. In contrast, with higher fluid flow rates, as shown in Figure S11(b), the bubbles experienced significant movement with breaking and coalescence. This motion enables hydrates, originally shrouded by water, to encounter the gas phase, expediting hydrate dissociation in these regions, as shown in Figure S12. As marked by the yellow circles in Figure S11(b), hydrates that do not decompose at low flow rates undergo dissociation under high fluid flow rate conditions.

To elucidate the effects of the fluid flow, hydrate dissociation ratio at t=30 s in each block at the cross-section of x=1.4 mm is counted in Figure S11(c). When the fluid flow rate is low, the hydrate dissociation ratio is lower than 10% in nearly half of the blocks. This indicates that mass transfer limitation in water plays an important role. The heat map of the dissociation rate reveals that in the upper left region of the cross-section, the hydrate dissociation rate is higher due to the presence of several gas bubbles. Conversely, the lower-

right quadrant, predominantly enveloped by water, faces strong mass transfer limitation, resulting in minimal dissociation rates. The dissociation shows a non-uniform pattern. When the flow rate was higher, the number of blocks with dissociation rates below 10% decreased compared to the low fluid flow rate case. This indicates that bubble movement due to the fluid flow weakens the mass transfer limitation and promotes hydrate dissociation. As can be seen from the dissociation rate heat map, although the flow of bubbles increases the hydrate dissociation rate in general, hydrate dissociation is still concentrated in the upper left region. The cross-section of the phase distribution shows that the upper left part of the cross-section has a larger porosity and better permeability due to the hydrate dissociation, which forms dominant channels. Bubbles, consequently, preferentially navigate through these dominant channels, primarily enhancing hydrate dissociation therein. This focused dissociation in dominant channels augments their width, forming a wormholing dissociation pattern.

Figure 3D summarises the curves of hydrate saturation with time for different conditions. As the water saturation increases, the methane hydrate dissociation rate declines significantly due to the mass transfer limitation effect. For the same sediment water saturation, a higher fluid flow rate corresponds to a faster dissociation rate, reinforcing the mitigating role of bubble movement on mass transfer limitation. It is important to note that the difference in the rate of hydrate dissociation under high versus low fluid flow rate conditions is not as significant in the three-dimensional case compared to previous two-dimensional work.54 In authentic three-dimensional micro-CT images, the hydrate pore habit is intricately complex, and many rock surfaces remain uncoated by hydrate, potentially reducing the likelihood of moving bubbles contacting new hydrate formations. Concurrently, as mentioned above, given the heterogeneity of porosity and permeability in three-dimensional structures, bubbles predominantly flow through the dominant channels. The hydrate dissociation rate in these dominant channels is already high and most of the hydrates have decomposed. Thus, the bubbles flowing in

Figure 5. Regime analysis (A) Quantification of the heat and mass transport limitations under different water saturations. MTL represents the mass transfer limitation, HTL denotes the heat transfer limitation and MNDR is the dissociation rate under multiphase non-isothermal conditions. (B) Regime diagram of methane hydrate dissociation patterns. The dots in the figure correspond to the conditions of each set of numerical simulations. (C) Heat map of hydrate dissociation rate in each block at the cross section of x=1.4 mm for three typical dissociation patterns: uniform (Sw=0.20, Pe~O(100)), nonuniform (Sw=0.60, Pe~O(10−2)) and wormholing (Sw=0.60, Pe~O(100)).

![](<Yang 等 - 2024 - Three-dimensional pore-scale study of methane hydrate dissociation mechanisms based on micro-CT imag_images/imageFile8.png>)

temperature field of Figure 2, where the temperature difference within the domain do not exceed 0.1K. When the fluid flow rate is high, as shown in Figure S14, the temperature field trend closely mirrors that observed under low flow conditions, suggesting that the influence of flow on the heat transfer process is relatively insignificant.

Figure 4B illustrates the heat map of the hydrate dissociation rate distribution at x=1.4 mm cross-section. The overall hydrate dissociation rates are lower due to the decreased

these dominant channels have limited effect on the enhancement of hydrate dissociation rate. Contrarily, our prior two-dimensional simulations assumed an idealized grain-coating hydrate structure, where bubble movement invariably leads to exposure of water-encased hydrate to the gas phase, significantly enhancing hydrate dissociation. This discrepancy underlines the limitations of two-dimensional idealized models. There is a necessity for further corrections and improvements using a three-dimensional actual structure.

temperatures. The dissociation is mainly concentrated in the region encapsulated by the gas phase. When the flow rate increases, bubble motion gravitates towards the dominant channel, identified by the red triangle, facilitating greater hydrate dissociation in this area by mitigating mass transfer limitation. These phenomena are similar to those shown in Figure S11 for the higher water saturation case. The dissociation of hydrates exhibits nonuniform behaviour at low flow rates, correlated with gas-water distribution, whereas at high flow rates, there is a propensity for wormhole formation. As delineated above, a uniform temperature field characterizes the computational domain during non-isothermal dissociation. The alterations in temperature impact merely the apparent reaction rate and the terminal equilibrium state, without significantly modifying the hydrate dissociation rate distribution. Consequently, the dissociation pattern does not alter radically when considering the heat transfer effect.

###### Effect of heat transfer on the dissociation process

After modelling isothermal methane hydrate dissociation processes to identify the mass transport limitation mechanism, the thermal effect and competition mechanisms between heat and mass transport are investigated by taking the reaction heat into account in the non-isothermal simulation. Different water saturations and fluid flow rates are employed in the simulation. Figure 4A shows the numerical results with the water saturation of Sw=0.40 under low fluid flow rate (Pe~O(10−2)). There is a significant concentration gradient in the concentration field in the water phase, indicating that mass transfer limitation still plays an important role. The phase distribution shows that hydrates enveloped by the gas phase undergo rapid dissociation, while those encased in water remain largely intact. Unlike the isothermal process, the methane concentration in the water phase decreases generally as dissociation proceeds under non-isothermal conditions. This decline can be attributed to temperature reduction induced by the endothermic nature of the dissociation reaction, subsequently affecting hydrate dissociation equilibrium concentration. The hydrate surface dissociation reaction is inhibited, and less methane is produced, thus the concentration decreases. The accompanying temperature map highlights the significant temperature drop due to heat absorbed during dissociation, reaching nearly 282 K at 30 s. This reduced temperature hampers the dissociation rate, as dictated by the dissociation kinetics, which is regarded as ‘heat transfer limitation’. Although the sediment temperature experienced a significant decrease, the temperature field remains predominantly uniform across the reservoir. As shown in Figure S13, even at the moment of the fastest dissociation rate at the beginning (t=1 s), the maximum temperature variance within the computational domain is confined to less than 3 K. In later stages, this temperature disparity becomes even more marginal. Consequently, the uniform temperature distribution across the domain does not result in substantial regional variations in hydrate dissociation, indicating that heat transfer processes exert a negligible impact on hydrate structural evolution within a limited reservoir framework. The difference in the hydrate dissociation patterns is still mainly caused by the role of mass transfer limitation. Similar results are also reflected in the

ted in Figures 4C & D. Compared to Figure 3D, there is a marked reduction in hydrate dissociation rates when heat absorption from the dissociation reaction is incorporated, signifying a heat transfer limitation. The temperature variation curves reveal that the rate of temperature decline is inversely correlated with increasing reservoir water saturation. This is attributable to two primary factors. First, elevated water saturation imposes mass transfer limitations that impede hydrate dissociation, thereby reducing heat absorption. Second, it is due to the higher specific heat capacity of water than the gas phase. Consequently, lower water saturation accentuates the effects of heat transfer limitations. As a result, the difference between the two curves for water saturation Sw=0.20 and Sw=0.40 in Figure 4C is smaller compared to Figure 3D. The essence behind this is the result of the combined effect of heat transfer limitation and mass transfer limitation.

###### Regime analysis for the methane hydrate dissociation

Analysing the above numerical findings, we deduce that the rate of methane hydrate dissociation emerges from the interplay between heat and mass transfer limitations, with fluid flow introducing additional variances to the dissociation pattern and rate. To understand these mechanisms more clearly, we first quantified the heat and mass transfer limitation effects, as shown in Figure 5A. Only the low flow rate case is shown here, and the results are similar for the high fluid flow rate case. Three typical dissociation scenarios, single-gas-phase isothermal dissociation, multiphase isothermal dissociation with different water saturation, and multiphase non-isothermal dissociation with different water saturation are simulated to quantify the role of heat and mass transfer limitations. The hydrate dissociation rate at t=1 s is calculated as the evaluation indicator. The difference between the single-

energy

phase and multiphase results is used to quantify mass transfer limitation (MTL), while the difference between isothermal and non-isothermal results is used to quantify heat transfer limitation (HTL). In a reservoir scenario where a single-phase gas is present, the main limitation effect on hydrate dissociation is the heat transfer limitation effect caused by the temperature decrease. The heat transfer limitation effect reduces the hydrate dissociation rate by about 50%. With the increase of water saturation in the reservoir, the heat transfer limitation effect gradually decreases, and the mass transfer limitation effect gradually increases. When the water saturation reaches 0.60, the hydrate dissociation rate decreases by about 70% due to mass transfer limitation. Based on the water saturation of the sediment, a preliminary segmentation of the competition mechanisms of mass transfer limitation and heat transfer limitation effects has been carried out, resulting in three primary zones. When sediment water saturation is low (Sw<0.20), heat transfer limitation dominates. When the water saturation of the sediment is high (Sw>0.40), mass transfer limitation dominates. When the water saturation is moderate, the effects of heat transfer limitation and mass transfer limitation are comparable. This implies that tailored strategies are essential to enhance hydrate dissociation depending on sediment water saturations. For instance, in cases of high sediment water saturation, mitigation strategies like gas injection should be employed to counter the dominant mass transfer limitation effects. Conversely, when heat transfer limitations are significant, external heating measures should be considered to accelerate the hydrate dissociation process.

The fluid flow is a determinant factor in influencing the hydrate dissociation pattern, as derived from the above analysis. According to the numerical results with different water saturations and different fluid flow rates, we plot the regime diagram of the methane hydrate dissociation pattern, as shown in

- Figure 5B. The core disparity in these hydrate dissociation patterns stems from mass transfer limitations within water. The regime diagram can be first partitioned according to the level of water saturation in the sediment. As demonstrated in Figure 5A, when the sediment-water saturation is relatively


low (Sw<0.20), the mass transfer limitation effect is less influential. At this point most of the hydrate is encapsulated in the gas phase, and the dissociation rate is high throughout the sediment regardless of the flow rate. This scenario leads to a uniform dissociation pattern as shown in Figure 5C, where the hydrate dissociation ratio is high in most regions. When the water saturation of the sediment is high (Sw>0.20), the mass transfer limitation is more significant. The hydrate dissociation pattern is no longer uniform. In this case, the fluid flow influences the dissociation process. The regime diagram can be further partitioned according to the flow rate. At low fluid flow rates (Pe<O(100)), the capillary number Ca=ρwυwU/γ<O(10−3). In this condition, inertial forces are difficult to overcome the capillary forces in the pores, and the shifts in gas-water distribution remain negligible. Within the gas-phase and water-phase regions, a substantial disparity in hydrate dissociation rates is evident, as illustrated by the non-uniform dissociation pattern. In Figure 5C. When the fluid flow rate is higher (Pe >O(100), Ca >O(10−3)), the inertial forces dominate compared to the capillary forces. The bubbles start to move and predominantly flow through the dominant channel. The movement of gas in these channels diminishes the mass transfer limitation, promoting hydrate dissociation. As a result, the dominant channels expand, culminating in a wormholing dissociation pattern as shown in Figure 5C. Based on the above analysis, the partition boundary of the regime diagram can be determined by the water saturation and capillary number. Meanwhile, the results of dissociation patterns obtained from numerical simulations under different conditions fall within the corresponding intervals in Figure 5B, confirming the reasonableness of the zoning. The regime diagram delineating hydrate dissociation patterns offers valuable insights for anticipating changes in hydrate pore structures during dissociation, enhancing predictions related to transport processes within the sediment.

###### Upscaling parameters for the methane hydrate production forecast

After understanding the methane hydrate dissociation mechanisms and obtaining the regime diagram for the dissociation pattern, upscaling work can be conducted for the production forecast. The upscaling conception from the pore scale to the REV scale is presented graphically in Figure 1A. From a REVscale viewpoint, the hydrate field can be discretized into numerous control

volumes with each control volume containing a homogeneous mixture consisting of gas, water, sand, and hydrate. To obtain model parameters at the REV scale, detailed phase distribution information in the hydrate sediment porous media can first be obtained by pore-scale means such as microCT imaging. Relying on pore-scale numerical outcomes, dissociation mechanisms are pinpointed, and REV-scale modelling parameters are derived through upscaling analysis. As mentioned in Eqation (S30), in REV-scale models for predicting hydrate production, the flux term and the source term are critical components of the calculation process. Therefore, the permeability used for computing the flux term, and kinetic models for source term calculation based on Equation (S31), are the key parameters when simulating hydrate dissociation at the REV scale (Note S4).

The permeability model needs to consider the complex morphology of hydrate pore habit and the structural evolution during dissociation. To address this, we tracked the structural evolution of hydrates across various simulation conditions, determining permeability across different hydrate saturations, as depicted in Figure 1B. The normalized permeability is defined as the ratio of sediment permeability Kand the permeability in the absence of hydrate K0. Distinct permeability profiles emerge under high and low fluid flow rates. When the fluid flow rate is high, dominant channels are formed and widened in the sediment with high water saturation (Sw=0.40, 0.60) as discussed above. The continuous widening of the dominant channels promotes the fluid flow and therefore leads to higher permeability. When the sediment-water saturation is lower (Sw=0.20), increasing the fluid flow rate still improves permeability, but the increase is not as pronounced as with higher saturation. Although the dissociation pattern is uniform for both high and low fluid flow rate conditions, the flow of the fluid promotes the hydrate dissociation in the flow channel to some extent, improving the permeability.

The differences in permeability curves during hydrate decomposition at different flow rates illustrate the need for permeability model to consider the hydrate dissociation pattern. We fit the permeability results for high and low Peconditions separately to obtain the corresponding permeability models as

= {52S3

+1 for low Pe 60S3

h −14S2

h −3S

h

K/K

(1)

0

h −20S2

h −2S

+1 for high Pe

h

The permeability model is different from the results calculated in our previous work using a two-dimensional ideal structure.43 For example, at high fluid flow rates, both two-dimensional and three-dimensional simulations show wormholing dissociation patterns. Figure 1B presents permeability curves from two-dimensional scenarios, diverging significantly from those in the present 3D work with a relative error exceeding 50%. This disparity stems primarily from the contrast between the idealized two-dimensional hydrate structure and the realistic three-dimensional counterpart. It is also influenced by differences in dissociation evolution and heat and mass transfer mechanisms. In actual three-dimensional structures, the intricacy of flow channels is heightened, leading to a preference for bubble flow through wider channels and larger pores, diverging from ideal two-dimensional structures. Hydrate dissociation predominantly occurs in areas with larger pores, and the impact on flow channel expansion is less pronounced than in two-dimensional scenarios. Consequently, this leads to notable differences in outcomes between real-3D and idealized-2D models. Incorrect estimation of the hydrate permeability will bring errors in the calculation of the flux term, which will seriously affect the accuracy of the hydrate production prediction. The importance of upscaling studies employing a three-dimensional realistic structure is thus emphasized.

Besides the permeability model, the kinetic model that determines the hydrate dissociation rate is also significant for REV-scale modelling. The dissociation rate calculation should account for both heat and mass transfer limitations. When considering the heat transfer limitation, the above analyses indicate that the temperature field within a finite volume remains relatively uniform throughout the hydrate dissociation process. Thus, the heat transfer limitation effect can be considered by directly adopting a single-temperature model and calculating the change of the average temperature within each REV (Note S4). This is consistent with our previous study.54

T

ment parameter into the REV-scale equation (Equation S30), a kinetic model that considers the mass transfer limitation effect is obtained. In the present work, we will upscale the kinetic model along this idea. Figure 1C calculates the variation of the effective reaction surface area during the dissociation process with hydrate saturation for different sediment water saturation conditions. Given that the disparity in hydrate dissociation rates between high and low fluid flow rates is marginal, as illustrated in Figure 3D, our focus in Figure 1C is solely on the results at the low flow rate. The results at high flow rates are not significantly different from those at low flow rates. As the water saturation increases, the effective reaction surface area decreases, which can reflect the increase of the mass transfer limitation effect. Based on the results of numerical simulations, we fitted the effective reaction surface area as

= (

###### )A

(

)

2

1−S

h0 −S

max(0,S

h −a) S

w

A

(2)

ers

s0

1−S

h0 −a

h0

where Sh0 is the initial hydrate saturation and As0 is the initial hydrate surface area. acharacterises the hydrate saturation that corresponds to the dissociation of all hydrates encapsulated in the gas phase, which can be calculated as

a = 0.2(

) (3)

1−S

h0 −S

w

1−S

h0

Figure 1C presents the fitting lines. The fitting results can be incorporated into REV models (Note S4) to characterise the role of mass transfer limitations. Consequently, a kinetic model that encompasses both heat and mass transfer limitations is formulated. It should be noted the effective reaction surface area model deduced from simulations within the three-dimensional real sediment structure demonstrates significant divergence from models derived from prior two-dimensional studies. Figure 1C also plots the effective reaction surface area model obtained from the two-dimensional study with Sw=0.20, displaying segmented characteristics. The effective reactive surface area is essentially constant over a range of hydrate saturations. This constancy in effective reaction surface area stems from the ideal grain-coating structure employed in preceding two-dimensional studies, wherein the hydrate predominantly encapsulates the sand throughout the dissociation process, maintaining an essentially constant surface area. Contrarily, in actual three-dimensional structures, hydrate pore habits are intricate and diverge from the ideal grain-coating structure. Throughout the dissociation process, the surface of the hydrate-coated sand is continuously exposed, and thus the hydrate surface area decreases from the beginning of the dissociation. As shown in Figure 1C, a substantial discrepancy exists between the twoand three-dimensional results. Employing the model derived from the twodimensional idealized structure would result in inaccurate estimations of the hydrate dissociation rate during the production forecast. Therefore, it is imperative to incorporate the authentic hydrate pore habit and its evolutionary process during the upscaling procedure.

To validate the correctness of the kinetic model, we bring it into the REVscale governing equations (Note S4) to predict the hydrate dissociation rate. The calculation procedure can be found in our previous work.54 Figure 1D compares the predicted results with the numerical simulation results, which are in good agreement. The average relative error is 3%-13%, which shows that our proposed kinetic model is reasonable. If the effective reaction surface area model from the two-dimensional study is utilized for predictions, it results in substantial deviations from the numerical results, exhibiting a relative error exceeding 30%, as shown in Figure 1D.

Although the idea of upscaling in the present work follows the idea of previous two-dimensional study,43,54 the results obtained are significantly different. This underscores the importance of capturing authentic sediment structures and real hydrate pore habit evolution during the dissociation process in upscaling research. Thus, three-dimensional digital image-based numerical simulations are indispensable.

###### CONCLUSION

In this study, comprehensive pore-scale numerical investigations are conducted to analyse methane hydrate dissociation via depressurisation,

using real hydrate sediment pore structures derived from three-dimensional micro-CT images. Employing the lattice Boltzmann framework, the research comprehensively models the hydrate dissociation process, integrating aspects like multiphase flow, heat and mass transfer, and hydrate structure evolution. The comparison of experimental and numerical results, particularly of xenon hydrate dissociation, validated the numerical model and elucidated the impact of heat and mass transfer mechanisms. Key findings reveal that mass transfer limitation in water significantly influences the rate of methane hydrate dissociation. This effect varies with sediment water saturation, leading to different dissociation patterns: uniform in low water saturation and nonuniform in high saturation scenarios. Additionally, bubble movement in the fluid, especially under high flow conditions, was observed to alter the dissociation pattern, creating a wormholing effect.

Three-dimensional real sediment structures show that bubble flow aids in hydrate dissociation, but less significantly compared to idealized two-dimensional models. The study also highlights the influence of temperature changes due to endothermic reactions, identifying heat transfer limitations. Upscaling work leads to the development of revised permeability and kinetic models for REV-scale production forecasting, significantly differing from previous two-dimensional models. This study underscores the necessity of using actual sediment structures and hydrate pore habits for accurate gas hydrate modelling and forecasting.

###### REFERENCES

1.

Chen, J.M. (2021). Carbon neutrality: Toward a sustainable future. The Innovation 2:

100127. DOI: 10.1016/j.xinn.2021.100127.

2.

Wang, F., Harindintwali, J.D., Yuan, Z., et al. (2021). Technologies and perspectives for achieving carbon neutrality. The Innovation 2: 100180. DOI: 10.1016/j.xinn.2021. 100180.

3.

Wang, F., Harindintwali, J.D., Wei, K., et al. (2023). Climate change: Strategies for mitigation and adaptation. The Innovation Geoscience 1: 100015. DOI: 10.59717/j. xinn-geo.2023.100015.

4.

Sloan, E.D. (2003). Fundamental principles and applications of natural gas hydrates. Nature 426: 353−359. DOI: 10.1038/nature02135.

5.

Sloan, E.D. (1998). Gas Hydrates: Review of Physical/Chemical Properties. Energy & Fuels 12: 191−196. DOI: 10.1021/ef970164+.

6.

Klauda, J.B., and Sandler, S.I. (2005). Global Distribution of Methane Hydrate in Ocean Sediment. Energy & Fuels 19: 459−470. DOI: 10.1021/ef049798o.

7.

Wei, J., Fang, Y., Lu, H., et al. (2018). Distribution and characteristics of natural gas hydrates in the Shenhu Sea Area, South China Sea. Mar. Pet. Geol. 98: 622−628. DOI: 10.1016/j.marpetgeo.2018.07.028.

8.

Feng, J.-C., Yan, J., Wang, Y., et al. (2022). Methane mitigation: Learning from the natural marine environment. The Innovation 3: 100297. DOI: 10.1016/j.xinn.2022. 100297.

9.

Feng, J.-C., Tang, L., Xie, Y., et al. (2023). Offshore carbon sequestration: Renewable energy and multi-carbon transformations prompt greener future. The Innovation Geoscience 1: 100016. DOI: 10.59717/j.xinn-geo.2023.100016.

10.

Kan, J., Sun, Y., Dong, B., et al. (2021). Numerical simulation of gas production from permafrost hydrate deposits enhanced with CO2/N2 injection. Energy 221: 119919. DOI: 10.1016/j.energy.2021.119919.

11.

Lee, S.Y., and Holder, G.D. (2001). Methane hydrates potential as a future energy source. Fuel Process. Techn. 71:181-186. DOI: https://doi.org/10.1016/S0378-3820 (01)00145-X.

12.

Song, Y., Yang, L., Zhao, J., et al. (2014). The status of natural gas hydrate research in China: A review. Renew. Sustain. Energy. Rev. 31: 778−791. DOI: 10.1016/j.rser.2013. 12.025.

13.

Anderson, B.J., Kurihara, M., White, M.D., et al. (2011). Regional long-term production modeling from a single well test, Mount Elbert Gas Hydrate Stratigraphic Test Well, Alaska North Slope. Mar. Pet. Geol. 28: 493−501. DOI: 10.1016/j.marpetgeo.2010.01. 015.

14.

Egorov, A.V., Nigmatulin, R.I., and Rozhkov, A.N. (2016). Heat and mass transfer effects during displacement of deepwater methane hydrate to the surface of Lake Baikal. Geo-mar. Lett. 36: 215−222. DOI: 10.1007/s00367-016-0443-9.

15.

Liu, L., Shao, H., Fu, S., et al. (2016). Theoretical simulation of the evolution of methane hydrates in the case of Northern South China Sea since the last glacial maximum. Environ. Earth. Sci. 75. DOI: 10.1007/s12665-015-5125-9.

16.

Chen, L., Feng, Y., Kogawa, T., et al. (2018). Construction and simulation of reservoir scale layered model for production and utilization of methane hydrate: The case of Nankai Trough Japan. Energy 143: 128−140. DOI: 10.1016/j.energy.2017.10.108.

17.

Pang, W.X., Xu, W.Y., Sun, C.Y., et al. (2009). Methane hydrate dissociation experiment in a middle-sized quiescent reactor using thermal method. Fuel 88: 497−503. DOI: 10. 1016/j.fuel.2008.11.002.

Ji, C., Ahmadi, G., and Smith, D.H. (2001). Natural gas production from hydrate decomposition by depressurization. Chem. Eng. Sci. 56:5801-5814. DOI: https://doi. org/10.1016/S0009-2509(01)00265-2.

18.

###### energy

19.

Yuan, Q., Sun, C., Yang, X., et al. (2011). Gas Production from Methane-HydrateBearing Sands by Ethylene Glycol Injection Using a Three-Dimensional Reactor. Energy & Fuels 25: 3108−3115. DOI: 10.1021/ef200510e.

20.

Chong, Z.R., Yang, S.H.B., Babu, P., et al. (2016). Review of natural gas hydrates as an energy resource: Prospects and challenges. Appl. Energy 162: 1633−1652. DOI: 10. 1016/j.apenergy.2014.12.061.

21.

Yang, J., Dai, X., Xu, Q., et al. (2021). Pore-scale study of multicomponent multiphase heat and mass transfer mechanism during methane hydrate dissociation process. Chem. Eng. J. 423: 130206. DOI: 10.1016/j.cej.2021.130206.

22.

Yin, Z., Chong, Z.R., Tan, H.K., et al. (2016). Review of gas hydrate dissociation kinetic models for energy recovery. J. Nat. Gas Eng. 35: 1362−1387. DOI: 10.1016/j.jngse. 2016.04.050.

23.

Moridis, G.J., Silpngarmlert, S., Reagan, M.T., et al. (2011). Gas production from a cold, stratigraphically-bounded gas hydrate deposit at the Mount Elbert Gas Hydrate Stratigraphic Test Well, Alaska North Slope: Implications of uncertainties. Mar. Petrol. Geol. 28: 517−534. DOI: 10.1016/j.marpetgeo.2010.01.005.

24.

Yu, M., Li, W., Jiang, L., et al. (2018). Numerical study of gas production from methane hydrate deposits by depressurization at 274 K. Appl. Energy 227: 28−37. DOI: 10.1016 /j.apenergy.2017.10.013.

25.

Yin, Z., Moridis, G., Chong, Z.R., et al. (2018). Numerical analysis of experimental studies of methane hydrate dissociation induced by depressurization in a sandy porous medium. Appl. Energy 230: 444−459. DOI: 10.1016/j.apenergy.2018.08.115.

26.

Lei, L., Seol, Y., and Jarvis, K. (2018). Pore-Scale Visualization of Methane HydrateBearing Sediments With Micro-CT. Geophys. Res. Lett. 45: 5417−5426. DOI: 10.1029/ 2018gl078507.

27.

He, G., Luo, X., Zhang, H., et al. (2018). Pore-scale identification of multi-phase spatial distribution of hydrate bearing sediment. J. Geophys. Eng. 15: 2310−2317. DOI: 10. 1088/1742-2140/aaba10.

28.

Zhang, J., Zhang, N., Sun, X., et al. (2023). Pore-scale investigation on methane hydrate formation and plugging under gas–water flow conditions in a micromodel. Fuel 333. DOI: 10.1016/j.fuel.2022.126312.

29.

Muraoka, M., Yamamoto, Y., and Tenma, N. (2020). Simultaneous measurement of water permeability and methane hydrate pore habit using a two-dimensional glass micromodel. J. Nat. Gas Sci. Eng. 77: 103279. DOI: 10.1016/j.jngse.2020.103279.

30.

Chen, Y., Sun, B., Chen, L., et al. (2019). Simulation and Observation of Hydrate Phase Transition in Porous Medium via Microfluidic Application. Ind. Eng. Chem. Res. 58: 5071−5079. DOI: 10.1021/acs.iecr.9b00168.

31.

Almenningen, S., Iden, E., Fernø, M.A., et al. (2018). Salinity Effects on Pore-Scale Methane Gas Hydrate Dissociation. J. Geophy. Res. Solid Earth 123(7): 5599−5608. DOI: 10.1029/2017jb015345.

32.

Zhao, Z., and Zhou, X.P. (2020). Pore-scale effect on the hydrate variation and flow behaviors in microstructures using X-ray CT imaging. J. Hydrol. 584: 124678. DOI: 10. 1016/j.jhydrol.2020.124678.

33.

Kou, X., Feng, J.-C., Li, X.-S., et al. (2022). Visualization of interactions between depressurization-induced hydrate decomposition and heat/mass transfer. Energy 239. DOI: 10.1016/j.energy.2021.122230.

34.

Kou, X., Feng, J.-C., Li, X.-S., et al. (2022). Memory effect of gas hydrate: Influencing factors of hydrate reformation and dissociation behaviors. Appl. Energy 306. DOI: 10. 1016/j.apenergy.2021.118015.

35.

Lei, L., Seol, Y., Choi, J.-H., et al. (2019). Pore habit of methane hydrate and its evolution in sediment matrix – Laboratory visualization with phase-contrast microCT. Marine Petrol. Geology 104: 451−467. DOI: 10.1016/j.marpetgeo.2019.04.004.

36.

Chen, X., Verma, R., Espinoza, D.N., et al. (2018). Pore-Scale Determination of Gas Relative Permeability in Hydrate-Bearing Sediments Using X-Ray Computed MicroTomography and Lattice Boltzmann Method. Water Resour. Res. 54(1): 600−608. DOI: 10.1002/2017wr021851.

37.

Yang, L., Falenty, A., Chaouachi, M., et al. (2016). Synchrotron X-ray computed microtomography study on gas hydrate decomposition in a sedimentary matrix. Geochem. Geophy. Geosys. 17: 3717−3732. DOI: 10.1002/2016gc006521.

38.

Jarrar, Z.A., Alshibli, K.A., Al-Raoush, R.I., et al. (2020). 3D measurements of hydrate surface area during hydrate dissociation in porous media using dynamic 3D imaging. Fuel 265. DOI: 10.1016/j.fuel.2019.116978.

39.

Kou, X., Li, X.-S., Wang, Y., et al. (2022). Hydrate decomposition front within porous media under thermal stimulation and depressurization conditions: Macroscale to microscale. Int. J. Heat Mass Trans. 188. DOI: 10.1016/j.ijheatmasstransfer.2022. 122653.

40.

Zhang, L., Zhang, C., Zhang, K., et al. (2019). Pore‐Scale Investigation of Methane Hydrate Dissociation Using the Lattice Boltzmann Method. Water Resour. Res. 55: 8422−8444. DOI: 10.1029/2019wr025195.

41.

Wang, X., Dong, B., Chen, C., et al. (2019). Pore-scale investigation on the influences of mass-transfer-limitation on methane hydrate dissociation using depressurization. Int. J. Heat Mass Trans. 144: 118656. DOI: 10.1016/j.ijheatmasstransfer.2019.

118656. Wang, X., Dong, B., Wang, F., et al. (2019). Pore-scale investigations on the effects of ice formation/melting on methane hydrate dissociation using depressurization. Int. J. Heat Mass Trans. 131: 737−749. DOI: 10.1016/j.ijheatmasstransfer.2018.10.143.

42.

43.

Yang, J., Xu, Q., Liu, Z., et al. (2022). Pore-scale study of the multiphase methane hydrate dissociation dynamics and mechanisms in the sediment. Chem. Eng. J. 430: 132786. DOI: 10.1016/j.cej.2021.132786.

44.

Katagiri, J., Konno, Y., Yoneda, J., et al. (2017). Pore-scale modeling of flow in particle packs containing grain-coating and pore-filling hydrates: Verification of a Kozeny–Carman-based permeability reduction model. J. Nat. Gas Sci. Eng. 45: 537−551. DOI: 10.1016/j.jngse.2017.06.019.

45.

Singh, H., Mahabadi, N., Myshakin, E.M., et al. (2019). A Mechanistic Model for Relative Permeability of Gas and Water Flow in Hydrate‐Bearing Porous Media With Capillarity. Water Resour. Res. 55(4): 3414−3432. DOI: 10.1029/2018wr024278.

46.

Hou, J., Ji, Y., Zhou, K., et al. (2018). Effect of hydrate on permeability in porous media: Pore-scale micro-simulation. Int. J. Heat Mass Trans. 126: 416−424. DOI: 10.1016/j. ijheatmasstransfer.2018.05.156.

47.

Wang, J., Zhang, L., Ge, K., et al. (2020). Characterizing anisotropy changes in the permeability of hydrate sediment. Energy 205: 117997. DOI: 10.1016/j.energy.2020. 117997.

48.

Zhang, L., Ge, K., Wang, J., et al. (2020). Pore-scale investigation of permeability evolution during hydrate formation using a pore network model based on X-ray CT. Marine Petrol. Geology 113: 104157. DOI: 10.1016/j.marpetgeo.2019.104157.

49.

Li, P., Zhang, X., and Lu, X. (2019). Three-dimensional Eulerian modeling of gas–liquid–solid flow with gas hydrate dissociation in a vertical pipe. Chem. Eng. Sci. 196: 145−165. DOI: 10.1016/j.ces.2018.10.053.

50.

Feng, Y., Chen, L., Suzuki, A., et al. (2019). Numerical analysis of gas production from layered methane hydrate reservoirs by depressurization. Energy 166: 1106−1119. DOI: 10.1016/j.energy.2018.10.184.

51.

Ren, X., Guo, Z., Ning, F., et al. (2020). Permeability of hydrate-bearing sediments. Earth Sci. Rev. 202: 103100. DOI: 10.1016/j.earscirev.2020.103100.

52.

Wu, D., Li, S., Guo, Y., et al. (2023). A novel model of reaction specific surface area via depressurization and thermal stimulation integrating hydrate pore morphology evolution in porous media. Chem. Eng. J. 452. DOI: 10.1016/j.cej.2022.139097.

53.

Yu, P.-Y., Sean, W.-Y., Yeh, R.-Y., et al. (2017). Direct numerical simulation of methane hydrate dissociation in pore-scale flow by using CFD method. Int. J. Heat Mass Trans. 113: 176−183. DOI: 10.1016/j.ijheatmasstransfer.2017.05.053.

54.

Yang, J., Xu, Q., Liu, Z., et al. (2023). Upscaling methane hydrate dissociation kinetic model during depressurisation. Chem. Eng. Sci. 275. DOI: 10.1016/j.ces.2023.118742.

55.

Latt, J., Malaspinas, O., Kontaxakis, D., et al. (2021). Palabos: Parallel Lattice Boltzmann Solver. Comput. Math. Appl. 81: 334−350. DOI: 10.1016/j.camwa.2020.03. 022.

56.

Li, Q., Luo, K.H., Kang, Q.J., et al. (2016). Lattice Boltzmann methods for multiphase flow and phase-change heat transfer. Prog. Energy Combust. Sci. 52: 62−105. DOI: 10.1016/j.pecs.2015.10.001.

57.

Chen, L., He, A., Zhao, J., et al. (2022). Pore-scale modeling of complex transport phenomena in porous media. Prog. Energy Combust. Sci. 88. DOI: 10.1016/j.pecs. 2021.100968.

58.

Chen, X., Espinoza, D.N., Luo, J.S., et al. (2020). Pore-scale evidence of ion exclusion during methane hydrate growth and evolution of hydrate pore-habit in sandy sediments. Marine Petrol. Geology 117: 104340. DOI: 10.1016/j.marpetgeo.2020. 104340.

###### FUNDING AND ACKNOWLEDGMENTS

The research is supported by the UK Engineering and Physical Sciences Research Council (EPSRC) under the grant No. EP/W026260/1. ARCHER2 supercomputing resources provided by the EPSRC project “UK Consortium on Mesoscale Engineering Sciences (UKCOMES)” (Grant No. EP/X035875/1) are gratefully acknowledged. This work made use of computational support by CoSeC, the Computational Science Centre for Research Communities, through UKCOMES. This work is also supported by the National Natural Science Foundation of China (No. 52206014). The funders had no role in study design, data collection and analysis, decision to publish or preparation of the manuscript.

###### AUTHOR CONTRIBUTIONS

Junyu Yang and Qianghui Xu designed and performed the research. Junyu Yang, Qianghui Xu, Xuan Kou, Geng Wang, Timan Lei, Yi Wang and Xiaosen Li analyzed the data and participated in drafting the paper. Kai H. Luo and Qianghui Xu designed and supervised the research, secured the funds, and finalized the manuscript.

###### DECLARATION OF INTERESTS

Xiaosen Li is an Editorial Board member of The Innovation Energy and was blinded from reviewing or making final decisions on the manuscript. Peer review was handled

independently of this member and their research group. The other authors declare no conflicts of interest.

###### DATA AND CODE AVAILABILITY

Data are available from the corresponding author upon reasonable request.

SUPPLEMENTAL INFORMATION

It can be found online at https://doi.org/10.59717/j.xinn-energy.2024.100015

###### LEAD CONTACT WEBSITE

Qianghui Xu: https://me.bit.edu.cn/szdw/jsml/rnydlgcx/zlydwgcyjs/zjjqtjgry/7ebdd745 cddd45819b68c9aa1ce5da69.htm

Kai H. Luo: https://profiles.ucl.ac.uk/39937-kai-luo

The Innovation Energy, Volume 1

###### Supplemental Information Three-dimensional pore-scale study of methane hydrate dissociation mechanisms based on micro-CT images DOI：https://doi.org/10.59717/j.xinn-energy.2024.100015

Junyu Yang, Qianghui Xu, Xuan Kou, Geng Wang, Timan Lei, Yi Wang, Xiaosen Li, Kai H. Luo

###### Table of Contents

- Supplemental Note S1. Governing equations and numerical models
- Supplemental Note S2. Comparison of experimental and numerical results
- Supplemental Note S3. Numerical settings for mechanism analysis
- Supplemental Note S4. Upscaling conception
- Supplemental Note S5. Numerical details and verification


- Figure S1. Schematic diagram of multiphysical processes during methane hydrate dissociation.
- Figure S2. Schematic diagram of the overall numerical implementation.
- Figure S3. Flow chart for the comparison between experimental and numerical results of xenon hydrate dissociation.
- Figure S4. Micro-CT images of the methane hydrate sediment with different water saturation for the numerical simulation.
- Figure S5. Schematic diagram of the selection of adjacent fluid node in the wetting boundary treatment.
- Figure S6. (a) The numerical result of a droplet on a spherical solid particle in three-dimensional view. (b) Numerical results in y-z cross section.
- Figure S7. (a) Schematic representation of the computational domain for the multiphase reactive transport between two plates. (b) Comparison of the numerical results and the analytical solution.
- Figure S8. (a) Schematic representation of the computational domain for the conjugate heat transfer between two plates. (b) Comparison of the numerical results and the analytical solution.
- Figure S9. The average concentration curves in gas phase obtained by the simulation with different mesh sizes.
- Figure S10. Enlarged view of methane concentration in water at t=100 s in the simulation of xenon hydrate dissociation experiment. The methane concentration is high near the hydrate surface, limiting the dissociation rate.
- Figure S11. Numerical results of isothermal methane hydrate dissociation. (a) Numerical results with


Sw 0.60 , PeO102 , including the phase distribution and concentration in water Cw . (b) Numerical results with Sw 0.60 , PeO100 . (c) Histograms and heat maps of hydrate

dissociation rate in each block at the cross section of x  0.14 mm .

- Figure S12. Comparison of phase distribution at t 1 s and t  30 s in the cross section of

x1.4 mm withtheconditionof Sw 0.60, PeO100 toidentifythebubblemovement.

- Figure S13. Enlarged view of temperature distribution at t 1 s in the numerical results of

non-isothermalmethanehydratedissociationwith Sw 0.40, PeO102.Thedifferencebetween

the highest and lowest values of temperature is not more than 3 K.

- Figure S14. Numerical results of non-isothermal methane hydrate dissociation with Sw  0.40 ,


PeO102,includingthephasedistribution,concentrationinwater Cw andtemperature T .

- Table S1. Physical properties for the simulation of xenon hydrate dissociation.
- Table S2. Physical properties for the simulation of methane hydrate dissociation.
- Table S3. Selection criteria for the adjacent fluid node


###### Supplemental Note S1. Governing equations and numerical models

- S1.1. Governing equations and assumptions Methane hydrate dissociation induced by depressurisation is investigated in the present work,


which involves a series of multiphysical processes in the pore structure as shown in Figure S1. When dissociation occurs on the hydrate surface due to the disruption of thermodynamic equilibrium, gas and water are generated associated with hydrate melting (Figure S1(c)), leading to the reconfiguration of phase distribution within the sediment (Figure S1(a)-(b)). On the water-covered hydrate surfaces, methane molecules, once released, need to traverse the water barrier into the gas phase via interfacial mass transfer (Figure S1(d)). The multiphase flow and interfacial mass transfer are fully coupled and substantially contribute to the progression of methane fugacity (or concentration). Simultaneously, the endothermic nature of the hydrate dissociation process absorbs heat from the reaction, resulting in a decrease in temperature via conjugate heat transfer (Figure S1(e)). These multiphysics phenomena alter the thermodynamic attributes of the sediment, thereby influencing the characteristics of the dissociation reaction. To formulate a mathematical representation of the above multiphysical processes, the present study incorporates several basic assumptions and simplifications, as follows:

- (1) Given the high reservoir pressure in the hydrate sediment (typically higher than 10 MPa),1 the

fluid flow of both water and gas phases are considered incompressible. Due to the limited free mean path of methane  at high pressure (1011 m ), the Knudsen number is very small ( Kn  0.001) in the micrometre-scale pore structure. Therefore, the non-slip boundary is assumed at the solid wall.

- (2) Methane molecules dissolved in the gas and water phase are treated as ideal solution. The

methane diffusion is governed by the Fick’s law and the concentration jump at the phase interface is described by Henry’s law.

- (3) Dissociation reaction only occurs at the hydrate surface, with no dissociation taking place

within the hydrate internal.2

- (4) Since the hydrate dissociation occurs at relatively low temperature in the present work (below


288.15 K), the physical properties can be regarded as constant because the temperature does not change drastically.

According to the above assumptions, governing equations of multiphase fluid flow, methane mass transfer and conjugate heat transfer can be written as

u  0 (S1)

 

            



      c b

u

p T t

 

uu u u F F (S2)

  



C

######    

      

C D C t

u (S3)

in w or g





T c c T T S t

##    

  

    

u (S4)

p p d



where . u . is the fluid velocity,  the fluid density, t the time, p the pressure,  the kinematic viscosity, Fs the capillary force at the gas-water interface and Fb the body force. C represents the methane concentration in water phase w and gas phase g , and D is the diffusivity. T is the temperature with cp and  being specific heat capacity and thermal conductivity of each phase. Sd is the heat absorption from the dissociation reaction. At the gas-water interface, species conservation reads3

 w  w w  g   g g 

C D C C D C C HC          

u w n u w n

(S5)

w g

The subscript w and g denote the properties in the water and gas phase. w is the velocity of the gas-water interface, n is the normal vector to the interface and H is the Henry coefficient. At the hydrate surface, dissociation occurs as

CH4 hH2O  CH4 hH2O, H  0 (S6)

where h is the hydrate number and H is the reaction enthalpy. The reaction mass flux Π at the hydrate surface nh can be calculated using the Kim-Bishnoi model as2

     

E k f f RT

0exp A  eq  h

Π n (S7)

 

where k0 is the pre-exponential factor, EA is the activation energy. f and feq are the local fugacity and equilibrium fugacity of methane, respectively. Based on the ideal solution assumption,

f can be replaced by the methane concentration as f  Cg  Cw / H . At each phase interface, conjugate heat transfer is considered by4

 T cp T  T cp T

######      

       

n u n u

 

T T

(S8)

where n is normal to the interface. The sign + and – denote parameters on either side of the interface. By this point, the governing equations - provide a comprehensive description of the multiphysical mechanisms inherent to the methane hydrate dissociation process. By introducing the characteristic length L , velocity U , density ch , concentration Cch and temperature Tch , dimensionless parameters marked by asterisks can be dereived by dimensionless analysis as

x y z tU x y z t

u u

    

- * * * * *

2

- * * * * c c


, , , , ,

L L L L U C T L



F



   

F (S9)

, , , ,

C T C T

 

ch ch ch

   

U UL k L c Ca Pe Da Pr

ch 0 ch

, , , p

   

D D

where key characteristic numbers are capillary number Ca , Péclet number Pe , Damköhler number Da and Prandtl number Pr .

- S1.2. Numerical models To effectively solve the above governing equations, pore-scale lattice Boltzmann (LB) models


are developed to simulate the hydrate dissociation process. The detailed implementation of these models is introduced in this section.

- S1.2.1. Phase field multiphase LB model In the present work, the phase field LB model5 is used to capture the gas-water multiphase flow


pattern. Beyond the solution of the Navier-Stokes (N-S) equations -, the order parameter , governed by the advection-diffusion type equation, is calculated in the phase field LB method to capture the phase interface. Two types of advection-diffusion equations are commonly used for the interface capture, namely the Cahn-Hilliard (CH)6 and Allen-Cahn (AC)7 equations. The CH equation initially gained more attention due to the lack of mass conservation in the AC equation.8-10 Nevertheless, the demand for higher order differentials and non-local numerical implementations limits the utility of the CH equation.11,12 To address these challenges, a conservative AC equation was proposed,13,14 which has been continuously developed and widely adopted in recent years. In the present work, an improved conservative AC equation is adopted as follows15:

                          

   

1 2 1 1 tanh ln

 

u n (S10)

M t W



2 1

1 indicates water phase and  0 gas phase. M is the mobility and W is the interface thickness scale parameter. The interface normal vector n  /  . To solve Eq. , the LB equation with the multiple-relaxation-time (MRT) collision operator16 using the D3Q19 lattice model17 is written as

                 

Λ f x f x M Λ m x m x I R

####     1    eq 

      

 

, , , ,

t t t t t

2 ( , ) ( , )



    

f t t t f t

x e x

  

, ,

###### (S11)

where f f,0, f,1 ,  , f,18 T istheorderparameterdistributionfunctionatposition x andtime

t . e is the discrete velocity along the αth direction. M is the transformation matrix connecting the velocity space and moment space as m  Mf. Λ is the diagonal relaxation matrix containing the parameters related to the mobility M . meq is the equilibrium moment and R represents the source term to calculate the final term on the right-hand side of Eq. . The detailed formulations of the parameters in Eq. can be found in Note S5. The order parameter can be obtained by

 (S12)

f,



and the fluid density is calculated by

g w g (S13)

where the constants w and g are the pure fluid densities in the water and gas phases, respectively.

Besides the AC equation, the incompressible N-S equations - also needs to be solved where the capillary force Fc can be calculated by the order parameter as

  

Fc  41  0.5   (S14)

2

where 12/W ,  3W / 2 and  is the surface tension. The MRT-LB equation for the incompressible fluid flow can be written as18-20

                 

Λ f x f x M Λ m x m x I R

 ,   ,  1   ,  eq  , 

 

t t t t t

2 ( , ) ( , )



    

f  t t t f t

x e x

(S15)

######  

f  f0, f1 ,  , f18 T is the pressure distribution function, Λ is the diagonal relaxation matrix related to the fluid viscosity, and R is the source term to involve the total force

 2 F   p  cs  Fc  Fb where cs  x / 3t stands for the lattice sound speed. The fluid pressure and velocity are calculated as

#####  

f

e

  

      

t t p f c

 

2

u u F (S16)

, 2 2 s s 2

 

c

More details of the parameter definitions can be found in Note S5. With the Chapman-Enskog analysis, the lattice Boltzmann equations and can recover the N-S equations and AC equation , and , and the LB models have been rigorously validated for the simulation of multiphase flow with high density ratios.

When simulating multiphase flow in the porous media, wetting boundary implementation at the

solid wall with accuracy and efficiency is a crucial issue in the phase field model. In general, there are two main approaches of wetting boundary treatment in the phase field framework, namely the surface-energy scheme21 and geometrical scheme.22-24 Since the geometrical scheme requires sufficient nodes for interface construction, its implementation complexity in three-dimensional complicated pore structure is unacceptable.25 Therefore, the surface-energy scheme is employed in the present model to realize the target contact angle , where the unknown order parameters in the solid nodes adjacent to the fluid s can be calculated by26

              

2

     

2 2 1 1 2 , cos 90 2 2

a a

   

o s f f sf

a a l a

   

  

o s f

90

(S17)

f is the order parameter of the adjacent fluid node at the direction of the solid surface normal vector ns . ns  s / s where the solid indicator s 1 for solid nodes and s  0 for fluid nodes. More details about the selection of adjacent fluid nodes and verification of the wetting boundary treatments can be found in Note S5.

- S1.2.2. Continuum species transport LB model for interfacial mass transfer To calculate the evolutionary pattern of solute concentrations in the multiphase solvent system,


two major types of LB implementations are commonly applied, namely the interface tracking scheme and the phase fraction indicating scheme.27 In the interface tracking scheme, the phase information of each fluid node is initially identified, followed by individual phase domain mass transfer calculations.28-30 At the phase interface, boundary treatment is adopted to achieve the interfacial mass transfer in compliance with Eq. . This treatment can handle interfacial mass transfer directly at the sharp interface. However, due to the requirement of consistently capturing interface positions for boundary treatment enforcement, this scheme is confined to closed systems where the interface does not undergo severe deformations. In the phase fraction indicating scheme,31,32 the phase interface is captured automatically based on the indicator parameters like the order parameter. Additional term is incorporated to realize the desired concentration profile at the interface according to the phase fraction distribution. Unlike the interface tracking scheme that necessitates artificially determined phase distribution information, the phase fraction indicating scheme can handle the interfacial mass transfer in the open systems integrating advection effects. Several recent efforts have contributed to the development of the phase fraction indicating scheme based on the LB method.33-35 Drawing inspiration from the continuum species transport (CST) model in the volume-of-fluid (VOF) framework, the CST-LB model is proposed in our previous work as a phase fraction indicating

scheme and has been successfully utilized to investigate interfacial mass transfer during the methane hydrate dissociation. The conventional CST model can be written as36

   

                

C D H C D C C t H

1 1

 

  

u (S18)

where the last term on the right-hand side is the CST source term realizing the concentration profile at the phase interface with species conservation as described by Eq. . The concentration and diffusivity are the phase-average values  

C Cw  1 Cg ,  

D  DwDg / Dw  1 Dg  . To solve this equation, the D3Q7 MRT CST-LB model,37 which is discretised on the first seven velocity directions of D3Q19 lattice model without significant accuracy loss, is employed as

                 

Λ g x g x N Λ n x n x Ι R

        

 

g

1 eq

, , , , CST

t t g g t g t t

2 , ,

   



    

g  t t t g t

x e x

(S19)

where  

g  g0, g1 ,  , g6 T is the concentration distribution function, N is the transformation matrix with ng  Ng , neqg is the equilibrium distribution function and Λg is the diagonal relaxation matrix related to the diffusivity. More details about the numerical model and validation can be found in Note S5. RCST is the CST additional term

H T C

               

    

1 1

R (S20)

0, , , , 0, 0, 0 4 (1 )

CST

H x y z

The concentration can be calculated by

 (S21)

C g



Using the Chapman-Enskog analysis, the LB equation can recover the CST advection-diffusion equation .

On the hydrate surface, the wall mass flux should follow the reaction mass flux as

######    

       

D H D C C

1 1

  

Π (S22)

H

In accordance with this relationship, dissociation reaction boundary treatment for the CST-LB model is given by27

######    

eg* x,t  Π eg* x  et,t (S23)

to calculate the unknown post-collision concentration distribution function in the solid nodes at

position x adjacent to the fluid node at x et , where  and  are in the opposite lattice direction.

- S1.2.3. Conjugate heat transfer LB model Different thermal LB models have been developed to solve the energy equations including the

multispeed scheme, double distribution function scheme and hybrid scheme.38,39 Among these methods, the double distribution function scheme4 is more suitable in the present work because it can accurately solve the advection-diffusion equation without using complicated high-order terms that multispeed scheme needs.39 The LB equation with D3Q7 lattice model can be written as

        

   

1 eq

, , , , d c , , t t h t h t t t h  t t t h t

 



             

h x h x N Λ Nh x n x R R x e x

(S24)

 

h  h0, h1 ,  , h6 T is the concentration distribution function, Λh is the diagonal relaxation matrix related to the thermal diffusivity and neqh is the equilibrium distribution function. The reaction heat source term at the hydrate surface Rd and conjugate source term Rc can be written as

d d d d h

- 3

, 0, 0, 0, , 0, 0 ,

- 4


T

s p p

S S

S A H

c c

      

   

R Π n (S25)

1  eq

c c c c

- 3 1 1

, 0, 0, 0, , 0, 0 , 1

- 4 2


T

p p

S S S c s h h T c     





                   

R  e u (S26)

where As is the hydrate surface area in each node. The temperature can be calculated by

T h



 (S27)

The LB equation is equivalent to the conjugate heat transfer equation and based on Chapman-Enskog analysis. More details of the numerical model and validations can be found in Note S5.

- S1.2.4. Hydrate structure update In the present work, the volume of pixel (VOP) method40 is applied to trace the morphological


evolution of hydrate structures, accomplished by updating the hydrate volume of each node. Initially, the hydrate volume Vh of the fluid node is zero. When the dissociation reaction proceeds, the hydrate volume in the hydrate nodes can be calculated by

######    

Vh t  t Vh t  ΠnhAsVm (S28)

where Vm is the molar volume of the hydrate. When the hydrate volume decreases to zero, the

hydrate node is converted into a fluid node. In the present work, it is assumed that the generated water occupies the region previously held by the decomposed hydrate and the methane concentration there equals to the equilibrium value. Therefore, in the newly converted fluid nodes, the physical properties are updated as 1, C  Ceq .

The overall numerical implementation is schematically demonstrated in Figure S2. Gas-water multiphase flow is first simulated to provide the phase distribution and velocity for the computation of heat and mass transfer modules. The reaction rate calculated in the mass transfer model provides the dissociation rate and heat absorption source term for the solid update and heat transfer computation. In turn, the equilibrium concentration and dissociation kinetics used for the mass transfer model are determined using the temperature obtained from the heat transfer model. Considering the moderate hydrate dissociation rate, each numerical modules can be solved independently without intense coupling. Therefore, the present numerical framework eliminates the necessity for internal iterations. The numerical procedures are developed in-house with C++ and parallel programming based on message passing interface (MPI) is conducted to improve the computational efficiency.

- Supplemental Note S2. Comparison of experimental and numerical results We first compare the numerical simulation results with the micro-CT images from xenon hydrate


dissociation experiments to substantial the credibility of the numerical models and to elucidate the heat and mass transfer mechanisms underlying the experimental observations of the hydrate dissociation. The experiment was conducted by Kou et al..41 Following the formation of xenon hydrates at 0.28 MPa, 278.65 K, a stop-and-go depressurization technique was employed to trigger the hydrate dissociation. The evolution of hydrate structure was captured by micro-CT imaging, and the hydrate structures at the early and late stage of the dissociation with hydrate saturation Sh  0.22 and Sh  0.03 are shown in Figure S3(a) and Figure S3(d). As the dissociation proceeds, the hydrate surface adjacent to pores gradually recedes into a concave surface, and the patchy hydrate cluster (Figure S3(a)) eventually evolves into the load-bridging pattern (Figure S3(d)) among the sand grains. It was found in the experiment that the meniscus surface of hydrate is closely related to the curved water layer enveloping the hydrate. Regarding the role of water, the researchers attributed the characteristics of hydrate structure evolution to the mass transfer limitation in water, considering that the impact of heat transfer between water and hydrate is inconsequential.

Due to the limited spatial and temporal resolution, the effects of heat and mass transfer mechanisms on hydrate dissociation pattern could not be comprehensively captured. For better understanding of the hydrate dissociation process, the numerical simulation reproduces the experimental procedures, as illustrated in Figure S3. The micro-CT image with the size of 2 mm2 mm2 mm is segmented to serve as the computational domain for the numerical simulation.

The simulation yields key parameters such as concentration and temperature evolution, aiding in the analysis of heat and mass transfer mechanisms. Simultaneously, dynamic patterns of hydrate structure evolution are also captured in the simulation. The numerical results (Figure S3(c)) of the post-dissociation hydrate structure will be compared with the experimental data (Figure S3(d)) to verify the reliability of the numerical models.

In the numerical simulation, the initial phase distribution of water, gas and hydrate is obtained through image segmentation of micro-CT image before dissociation ( Sh  0.22 ). The initial temperature aligns with the experimental value T0  278.65 K , and the initial concentration in the gas phase is calculated based on the experiment condition that

######  

Cg0  C xenon@0.28 MPa  0.124 mol/L . The concentration in water remains in equilibrium with gas according to Henry’s law Cw0  HCg0 . At the boundary of the computational domain, Dirichlet boundary treatment is adopted for temperature and concentration as T  278.65 K , Cg  0.022 mol/L , which is consistent with the experimental conditions. It should be noted that, unlike the stop-and-go conditions in the experiment, the numerical simulation employs constant concentration and temperature conditions, thus eliminating the extended time at the stop stage. As for the gas-water multiphase flow, the no-flux condition for the order parameter and constant pressure condition p  p0 are imposed at the boundaries. The lattice grid size of x 105 m is adopted according to the mesh convergence study in Note S5. In the present work, we assume the kinetic parameters of xenon are the same with methane hydrate due to their comparable molecular structure.42 To simulate the low solubility and diffusivity of xenon in water, we assign nominal values to the Henry coefficient and diffusion coefficient in water as H  0.1 , Dw 108 m2/s . Other physical properties are obtained or estimated based on the previous work43-46 as listed in Table S1.

- Supplemental Note S3. Numerical settings for mechanism analysis After comparing experimental and numerical results for xenon hydrate dissociation, we delved

deeper into the dissociation mechanism of methane hydrate using numerical simulations under varied conditions. The structure of the methane hydrate sediment is shown in Figure S4, which is obtained from micro-CT imaging by Chen et al..47 The size of the computational domain is

- 4 mm4 mm4 mm , which is meshed with lattice grid size of x 105 m . The average grain size of the sand is L  600 μm and the porosity of the sediment is 0.40 . The initial hydrate saturation is


Sh0  0.27 . In the simulation, different water saturation Sw is adopted with a stochastic gas-water distribution depicted in Figure S4, to explore the effects of water in the sediment. A periodic boundary condition is applied along the x-direction, with fluid flow induced by body forces. Different fluid flow rates are employed to discuss the effect of gas-water migration on hydrate dissociation. By monitoring dissociation rates and morphological changes of hydrates under these conditions, we aim

to elucidate methane hydrate dissociation mechanisms and establish a regime diagram.

In the simulation, the total pressure sets at 12 MPa to mimic the reservoir conditoin.1 The initial temperature is set as T 0  288.15K , which corresponds to an equilibrium methane concentration in the gas phase as Cg,eq  6.47 mol/L based on our previous model48 as

       

9005.5 C exp 33.12 mol/L T

######  

g, eq

(S29)

Initially, the methane concentration in the gas phase is set to Cg0  2.0 mol/L (corresponding to a partial pressure of 4.4 MPa) to simulate reservoir conditions triggering hydrate dissociation after depressurisation. The methane concentration in water is maintained as the equilibrium concentration Cw0  HCg,eq  0.65 mol/L . When the simulation starts, gas and water begin to flow due to body forces while hydrate is decomposed, changing the concentration and temperature conditions in the reservoir. Isothermal processes are first simulated to highlight the role of mass transfer. Subsequently, non-isothermal processes are simulated to assess the effect of heat transfer, with a no heat-flux condition set on the y and z axis boundaries. Physical properties of methane gas are detailed in Table S2 and other parameters align with values in Table S1.

- Supplemental Note S4. Upscaling conception After recognising the hydrate dissociation mechanisms, upscaling work is carried out to obtain


the modelling parameters for the REV-scale production forecast. The upscaling conception from the pore scale to REV scale is presented graphically in Figure 5(a). From an REV-scale viewpoint, the hydrate field can be discretized into numerous control volumes with each control volume containing a homogeneous mixture consisting of gas, water, sand, and hydrate. The mass and energy equations within a REV can be written as49

d

 Qn  (S30)

MdV dV SdV dt

where M , Q and S are the accumulation term, flux term and source term, respectively. More details of the REV-scale modelling can be found in our previous work.48 To obtain model parameters at the REV scale, detailed phase distribution information in the hydrate sediment porous media can first be obtained by pore-scale means such as micro-CT imaging. Based on the phase distribution information, physical modelling can be conducted to simulate the hydrate dissociation process. Relying on pore-scale numerical outcomes, dissociation mechanisms are pinpointed, and REV-scale modelling parameters are derived through upscaling analysis. These parameters can help to improve the accuracy of the production forecasts.

For hydrate production prediction, the permeability and kinetics model, which are closely related

to the flux and source term in the REV-scale modelling, are paramount. Given the intricate hydrate distribution within reservoirs, it is vital to thoroughly examine the impact of hydrate pore habits on seepage flow. As dissociation progresses, the hydrate pore habits evolve continuously, which further affects the reservoir seepage characteristics. Thus, when constructing the permeability model, the morphological trajectory of hydrate dissociation is essential. This trajectory hinges on the dissociation mechanism. Consequently, we employ pore-scale modelling across varying dissociation mechanisms to forecast hydrate pore habit transformations and derive the associated permeability models.

Regarding hydrate dissociation kinetics, a reliable model should fully consider the factors coupling heat and mass transport with intrinsic kinetics. In our previous work,48 the dissociation reaction rate which considers the effect of heat and mass transfer in REV-scale modelling can be calculated by

     

E r k F A f f RT

0exp A A s eq

 

(S31)

where volume-average temperature T determines the heat transfer effect. The volume-average fugacity f , the hydrate surface area As and its adjustment factor FA reflects the mass transfer effect. Among these parameters, the construction of the FA model is crucial for the accurate prediction of the reaction rate by considering the mass transfer effect. Our previous work48 determined FA by evaluating the effective reaction surface area FA  Aers / As0 , guided by mechanism insights. In this work, we will follow this idea to obtain a more reasonable kinetic model based on three-dimensional digital images.

###### Supplemental Note S5. Numerical details and verification

###### S5.1. Numerical parameters for D3Q19 MRT phase field LB model

For D3Q19 model, the discrete velocities e and the transformation matrix M are written as50,51

                     

0 1 1 0 0 0 0 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 1 1 0 0 1 1 1 1 0 0 0 0 1 1 1 1 0 0 0 0 0 1 1 0 0 0 0 1 1 1 1 1 1 1 1

e (S32)



                                                       

1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 1 1 0 0 0 0 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 1 1 0 0 1 1 1 1 0 0 0 0 1 1 1 1 0 0 0 0 0 1 1 0 0 0 0 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 1 1 1 1 0 0 0 0 0 0 0 0

    

         

 

 

- 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 0 0 0 0

- 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1
- 0 1 1 1 1 1 1 2 2 2 2 2 2 2 2 2 2 2 2


- 0 1 1 1 1 0 0 0 0 0 0 1 1 1 1 1 1 1 1


 

     

M  0 1 1 0 0 1 1 1 1 1 1 0 0 0 0 1 1 1 1

       

0 0 0 0 0 0 0 1 1 1 1 0 0 0 0 0 0 0 0

   

- 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 0 0 0 0
- 0 0 0 0 0 0 0 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 0 0 0 0


 

   

- 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1

- 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1
- 0 0 0 0 0 0 0 1 1 1 1 0 0 0 0 0 0 0 0


- 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0


 

0 0 0 0 1 1 1 1

(S33)

When solving the AC equation, the equilibrium moment meq and the source term R are given by

T x y z

######              

      

u u u  u c u c u c u c u c u c c c c

, , , , 0, 0, 0, , 0, 0, , , , , , , , ,

eq

m (S34)

2 2 2 2 2 2 4 4 4

x s x s y s z s y s z s s s s

T x t x y t y z z

 

          

######         

0, , , , 0, 0, 0, 0, 0, 0, , , , ( ) ( ) ( ) ( ( )) ( ( )) ( ( ,

F u F u F t u F u c F u c F u c F u c F u c F u c

  

, , ,

   



  

  

2 2 2 2 , , , ,

R (S35)

)) ( ( )) ( ( )) ( ( ))

, , , 0 0, 0



   

- x t x s x t x s y t y s z t z s
- y t y s z t z s


    

2 2 , ,





where u  u x, uy , uz  is the fluid velocity and

             



1 1 , , 1 tanh ln x y z 3 2 1 F F F     W

2 , , ,

F n is related to the source term in Eq. . The



diagonal relaxation matrix Λ contains the parameter relative to the mobility as

### Λdiag1,  ,  ,  , 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 (S36)

where

    

2

x M c

2 1 1 s 2

  

 t

(S37)

For the incompressible N-S equations, the equilibrium moment meq and the force source term R can be written by

T

 

      

2 2 2 2 2 2 2 2

p c u c u c u c u u c u u c u u p c

, | |

u

s x s y s z s x y s x z s y z s

             

         

         

       

2 2 2 2 2 2 4 4 4 4 4 4

u u c u u c c u c u c u c u c u c u c p u u u c p u u u c p u u u

, , , , , , , , 0.5 , 0.5 , 0.5

x y s x z s s x s x s y s z s y s z



eq

m

     

4 2 2 2 4 2 2 2

- s x y z s y z x
- s x z y


  



4 2 2 2

(S38)

                     

T

                                                              

     

  

2 2 2 2 2 2 2 , , , , ,

u

c F c F c F c u F c u F c c

, , , , ,

s b x s b y s b z s x b y y s y b x x s s

                 

   

2 2 2 2 2 2 , , , ,

u F c u F c c u F c u F c c c c u F c u F c c

, , 2 5 ,2 ,

x b z z s z b x x s s z b y y s y b z z s s

  

2 2 2 2 2 , ,

u F u

b s s x b x x s y b y y s s

    

 

    



  

   

2 , ,

2 2 2 2 2 2 2

2

u F c u F

c c u c u u c F c c

, 2 ,

, 2 2 2 2 2 2 2 2 2 2

x b x x s z b z

z s s y x s x y y s b x s s

       

               

         

u c u u c F c c u c u u c F c c u c u u c F c c u c u u c F c c u c u u c F

2 , 2 , 2 , 2 , 2

z x s x z z s b x s s x y s x y x s b y s s

, , 2 2 2 2 2 2 2 2 2 2



R

- x z s x z x s b z s s z y s y z z s b y s s
- y z s y z y s b z


, , 2 2 2

     

2 2

c c

,

,

s s

                 

        

2 2 2 2 2 2 , ,

u c F c u F c c u c c u c F c u F c c u c c u c F c u F c c u c c

2 2 , 2 2 , 2 2

- x s b x x s y b y y s s z z s s
- x s b x x s z b z z s s y y s s


2 2 2 2 2 2 , ,

2 2 2 2 2 2 , ,

y s b y y s z b z z s s x x s s

(S39)

The diagonal relaxation matrix Λ is given by

Λdiag1, 1, 1, 1,  ,  ,  ,  ,  ,  , 1, 1, 1, 1, 1, 1, 1, 1, 1 (S40)

where  is related to the fluid viscosity as

      

2

x c

2 1 1 s 2



  



 t

(S41)

and  can be adjusted to improve the numerical stability.

- S5.2. Wetting boundary treatment for three-dimensional phase field LB model When using surface-energy scheme for the wetting boundary treatment, the key issue is which


adjacent fluid node should be chosen to calculate Eq. . The selection scheme for the adjacent fluid node is shown in Figure S5. After calculating the solid surface normal vector ns , The projection of

ns on the xy-plane, ns' can be obtained. Then we can calculate the angle of ns' with the x-axis, 1 and the angle with ns , 2 . 1 and 2 can be used as a criterion to determine which adjacent fluid node is selected for the calculation. The selection criteria are listed in Table S3 and the number of adjacent fluid node is marked in Figure S5.

To verify the accuracy of the above treatment of the wetting boundary conditions, we simulated a droplet on a spherical solid particle. The contact angle was set to  60 and the simulation results are shown in Figure S6. It can be seen that the droplets end up stabilising at the solid surface at a contact angle of 60o, proving that the wetting boundary treatment used in this work is accurate.

###### S5.3. Numerical parameters and validation for heat and mass transfer model In D3Q7 MRT model, the transformation matrix is given by37

      

1 1 1 1 1 1 1

- 0 1 1 0 0 0 0 0 0 0 1 1 0 0

- 0 0 0 0 0 1 1 6 1 1 1 1 1 1

0 2 2 1 1 1 1

- 0 0 0 1 1 1 1




    

   

N (S42)

         

           

For the CST-LB model, the equilibrium moment neqg and diagonal relaxation matrix Λg are given by

T

      

eq 3

n (S43)

g C Cux Cuy Cuz C

, , , , , 0, 0 4

Λg diag1,D,D,D, 1, 1, 1 (S44)

where D is related to the diffusion coefficient D as

    

1 1 1 2 4 D 2

x

 

D





t

 

(S45)

For the conjugate heat transfer LB model, the equilibrium moment neqh and diagonal relaxation matrix Λh are given by

T

      

eq 3

n (S46)

h T Tux Tuy Tuz T

, , , , , 0, 0 4

### Λh diag1,,,,1,1,1 (S47)

###### where

   



1 1 2 4

x c  t

1 p 2

 



 

 



(S48)

To verify the accuracy of the CST-LB model, multiphase reactive transport between two plates is simulated. The physical problem is shown in Figure S7(a). A solute can be dissolved in both fluid A and fluid B with the Henry coefficient H  0.2 . The upper and lower plates can undergo a chemical reaction described as

   

C D k C C

 eq s

###### n n



s

(S49)

The kinetic coefficient k  2.5104 m/s . The diffusion coefficients in fluid A and B are

DA  DB  510 m/s . The equilibrium concentrations in fluid A and B are CA,eq  HCB,eq  0.2 mol/L . In the simulation, the heights of fluid A and fluid B are both L 100 μm . A periodic boundary condition is set in the x and y directions. Initially, the concentration in fluid A and fluid B is set as zero. As the reaction progresses, the concentration begins to rise in both fluid A and B with an analytical solution as

9

               





2sin

z L Dt C z t C C z L

  (S50)

    

2 B, eq B, eq 2 1

n

( , ) cos exp , sin cos 2sin

n n n n n n

             

L L z L Dt

 



        

    

2 A, eq A, eq 2 1

n

C z t C C z L

( , ) cos exp ,

n n n n n n

        

sin cos tan ,

L L kL



 

   

n n

Da Da D

 

Comparison of the numerical results and analytical solution at t  0.5 s is shown in Figure S7(b), which shows good agreement with relative error of 2.9%, indicating the numerical model is accurate.

In order to verify the accuracy of the conjugate heat transfer model, we simulated the conjugate heat transfer process of multiphase fluid between two flat plates, and the physical problem is shown in Figure S8(a). The temperatures of the upper and lower plates are Tu  0.0 , Tl 1.0 in lattice unit. The physical properties of fluid A and B are A  6.30 , cpA  0.07 , A  0.517 , B  0.70 , cpB  0.05 , B  0.022 in lattice units. The heights of the two fluids are both L 100 in lattice unit.

For this conjugate heat transfer problem, the analytical solution for the steady state can be written as

 





T T z T z T z L L

 

   

B u d

,

d

 

A B

 

           



T T z T z T z L L

 

A u d

2 ,

u

 

A B

(S51)

Figure S8(b) compare the numerical results and analytical solution, which shows good agreement with relative error of 0.8%.

- S5.4. Mesh convergence analysis Prior to numerical simulation, a mesh convergence analysis is performed to determine the


appropriate grid size. Three sets of grid sizes, x  20 μm , x 10 μm and x  5 μm are adopted in the simulation of xenon hydrate dissociation experiments. The average concentration in the gas phase is used as a criterion to evaluate the numerical results at different grid accuracies, which is shown in Figure S9. It can be seen that the numerical results with the grid of x  20 μm deviate significantly from those of the other two sets of grid precision. It indicates that the low grid precision of x  20 μm is not applicable to the study in this paper. While the two sets of numerical results with high precision have little difference, implying that the grid precision has converged. To compromise the computational resources, we adopt the grid size of x 10 μm for use in the subsequent numerical simulation study.

![](<Yang 等 - 2024 - Three-dimensional pore-scale study of methane hydrate dissociation mechanisms based on micro-CT imag_images/imageFile9.png>)

###### Figure S1. Schematic diagram of multiphysical processes during methane hydrate dissociation.

![](<Yang 等 - 2024 - Three-dimensional pore-scale study of methane hydrate dissociation mechanisms based on micro-CT imag_images/imageFile10.png>)

###### Figure S2. Schematic diagram of the overall numerical implementation.

![](<Yang 等 - 2024 - Three-dimensional pore-scale study of methane hydrate dissociation mechanisms based on micro-CT imag_images/imageFile11.png>)

###### Figure S3. Flow chart for the comparison between experimental and numerical results of xenon hydrate dissociation.

![](<Yang 等 - 2024 - Three-dimensional pore-scale study of methane hydrate dissociation mechanisms based on micro-CT imag_images/imageFile12.png>)

###### Figure S4. Micro-CT images of the methane hydrate sediment with different water saturation for the numerical simulation.

![](<Yang 等 - 2024 - Three-dimensional pore-scale study of methane hydrate dissociation mechanisms based on micro-CT imag_images/imageFile13.png>)

###### Figure S5. Schematic diagram of the selection of adjacent fluid node in the wetting boundary treatment.

![](<Yang 等 - 2024 - Three-dimensional pore-scale study of methane hydrate dissociation mechanisms based on micro-CT imag_images/imageFile14.png>)

###### Figure S6. (a) The numerical result of a droplet on a spherical solid particle in three-dimensional view. (b) Numerical results in y-z cross section.

![](<Yang 等 - 2024 - Three-dimensional pore-scale study of methane hydrate dissociation mechanisms based on micro-CT imag_images/imageFile15.png>)

###### Figure S7. (a) Schematic representation of the computational domain for the multiphase reactive transport between two plates. (b) Comparison of the numerical results and the analytical solution.

![](<Yang 等 - 2024 - Three-dimensional pore-scale study of methane hydrate dissociation mechanisms based on micro-CT imag_images/imageFile16.png>)

###### Figure S8. (a) Schematic representation of the computational domain for the conjugate heat transfer between two plates. (b) Comparison of the numerical results and the analytical solution.

![](<Yang 等 - 2024 - Three-dimensional pore-scale study of methane hydrate dissociation mechanisms based on micro-CT imag_images/imageFile17.png>)

###### Figure S9. The average concentration curves in gas phase obtained by the simulation with different mesh sizes.

![](<Yang 等 - 2024 - Three-dimensional pore-scale study of methane hydrate dissociation mechanisms based on micro-CT imag_images/imageFile18.png>)

###### Figure S10. Enlarged view of methane concentration in water at t=100 s in the simulation of xenon hydrate dissociation experiment. The methane concentration is high near the hydrate surface, limiting the dissociation rate.

![](<Yang 等 - 2024 - Three-dimensional pore-scale study of methane hydrate dissociation mechanisms based on micro-CT imag_images/imageFile19.png>)

- Figure S11. Numerical results of isothermal methane hydrate dissociation. (a) Numerical results


with Sw 0.60, PeO102, including the phase distribution and concentration in water Cw . (b) Numerical results with Sw 0.60 , PeO100 . (c) Histograms and heat maps of hydrate dissociation rate in each block at the cross section of x  0.14 mm .

![](<Yang 等 - 2024 - Three-dimensional pore-scale study of methane hydrate dissociation mechanisms based on micro-CT imag_images/imageFile20.png>)

- Figure S12. Comparison of phase distribution at t 1 s and t  30 s in the cross section of

x1.4 mm withtheconditionof Sw 0.60, PeO100 toidentifythebubblemovement.

![](<Yang 等 - 2024 - Three-dimensional pore-scale study of methane hydrate dissociation mechanisms based on micro-CT imag_images/imageFile21.png>)

- Figure S13. Enlarged view of temperature distribution at t 1 s in the numerical results of


non-isothermal methane hydrate dissociation with Sw 0.40 , PeO102 . The difference

between the highest and lowest values of temperature is not more than 3 K.

![](<Yang 等 - 2024 - Three-dimensional pore-scale study of methane hydrate dissociation mechanisms based on micro-CT imag_images/imageFile22.png>)

- Figure S14. Numerical results of non-isothermal methane hydrate dissociation with Sw  0.40 ,


PeO102,includingthephasedistribution,concentrationinwater Cw andtemperature T .

###### Table S1. Physical properties for the simulation of xenon hydrate dissociation.

|Water density, w|1000 kg/m3|
|---|---|
|Gas density, g|16 kg/m3|
|Kinematic viscosity of water, w|1.0106 m2/s|
|Kinematic viscosity of gas, g|1.7107 m2/s|
|Gas-water surface tension, |5106 kg/s2|
|Xenon diffusivity in gas, Dg|1.0105 m2/s|
|Pre-exponential factor, k0|5.13109 m/s|
|Activation energy, EA / R|9399 K|
|Reaction enthalpy, H|51.86 kJ/mol|
|Equilibrium concentration, Cg,eq|0.124 mol/L|
|Thermal conductivity of water, w|0.55 W/mK|
|Thermal conductivity of gas, g|0.005 W/mK|
|Thermal conductivity of hydrate, h|0.49 W/mK|
|Thermal conductivity of sand, s|0.90 W/mK|
|Specific heat capacity of water, cpw|4.2 kJ/(kgK)|
|Specific heat capacity of gas, cpg|0.16 kJ/(kgK)|
|Specific heat capacity of hydrate, cph|2.1103 kJ/(m3 K)|
|Specific heat capacity of sand, cps|2.0103 kJ/(m3 K)|


###### Table S2. Physical properties for the simulation of methane hydrate dissociation.

|Gas density, g|100 kg/m3|
|---|---|
|Kinematic viscosity of gas, g|1.7107 m2/s|
|Methane diffusivity in gas, Dg|1.0105 m2/s|
|Thermal conductivity of gas, g|0.045 W/mK|
|Specific heat capacity of gas, cpg|3.0 kJ/(kgK)|


###### Table S3. Selection criteria for the adjacent fluid node

|2|1|Number of adjacent fluid node|
|---|---|---|
|22.5 2  22.5|22.5 1  22.5|1|
| |22.5 1  67.5|7|
| |67.5 1 112.5|3|
| |112.5 1 157.5|8|
| |157.5 1 180,  180 1  157.5|2|
| |157.5 1  112.5|10|
| |112.5 1  67.5|4|
| |67.5 1  22.5|9|
|22.5 2  67.5|22.5 1  22.5|11|
| |22.5 1  67.5|19|
| |67.5 1 112.5|15|
| |112.5 1 157.5|20|
| |157.5 1 180,  180 1  157.5|12|
| |157.5 1  112.5|22|
| |112.5 1  67.5|16|
| |67.5 1  22.5|21|
|67.5 2  22.5|22.5 1  22.5|13|
| |22.5 1  67.5|23|
| |67.5 1 112.5|17|
| |112.5 1 157.5|24|


| |157.5 1 180,  180 1  157.5|14|
|---|---|---|
| |157.5 1  112.5|26|
| |112.5 1  67.5|18|
| |67.5 1  22.5|25|
|2  67.5|180 1 180|5|
|2  67.5|180 1 180|6|


###### References

- 1. Niu, M., Wu, G., Yin, Z., et al. (2021). Effectiveness of CO2-N2 injection for synergistic CH4 recovery and CO2 sequestration at marine gas hydrates condition. Chemical Engineering Journal 420:129615. DOI: https://doi.org/10.1016/j.cej.2021.129615.

- 2. Kim, H., Bishnoi, P.R., Heidemann, R.A., and Rizvi, S.S. (1987). Kinetics of methane hydrate decomposition. Chemical Engineering Science 42(7):1645-1653. DOI: https://doi.org/10.1016/0009-2509(87)80169-0.

- 3. Maes, J., and Soulaine, C. (2018). A new compressive scheme to simulate species transfer across fluid interfaces using the Volume-Of-Fluid method. Chemical Engineering Science 190:405-418. DOI: 10.1016/j.ces.2018.06.026.
- 4. Karani, H., and Huber, C. (2015). Lattice Boltzmann formulation for conjugate heat transfer in heterogeneous media. Physical Review E 91(2):023304. DOI: 10.1103/PhysRevE.91.023304.
- 5. Wang, H., Yuan, X., Liang, H., et al. (2019). A brief review of the phase-field-based lattice Boltzmann method for multiphase flows. Capillarity 2(3):33-52. DOI: 10.26804/capi.2019.03.01.
- 6. Cahn, J.W., and Hilliard, J.E. (1958). Free energy of a nonuniform system. I. Interfacial free energy. The Journal of chemical physics 28(2):258-267. DOI.
- 7. Allen, S.M., and Cahn, J.W. (1976). Mechanisms of phase transformations within the miscibility gap of Fe-rich Fe-Al alloys. Acta Metallurgica 24(5):425-437. DOI: https://doi.org/10.1016/0001-6160(76)90063-8.

- 8. Lee, T., and Lin, C.-L. (2005). A stable discretization of the lattice Boltzmann equation for simulation of incompressible two-phase flows at high density ratio. Journal of Computational Physics 206(1):16-47. DOI: 10.1016/j.jcp.2004.12.001.
- 9. Zheng, H.W., Shu, C., and Chew, Y.T. (2005). Lattice Boltzmann interface capturing method for incompressible flows. Phys Rev E Stat Nonlin Soft Matter Phys 72(5 Pt 2):056705. DOI: 10.1103/PhysRevE.72.056705.
- 10. Fakhari, A., and Rahimian, M.H. (2010). Phase-field modeling by the method of lattice Boltzmann equations. Phys Rev E Stat Nonlin Soft Matter Phys 81(3 Pt 2):036707. DOI: 10.1103/PhysRevE.81.036707.
- 11. Geier, M., Fakhari, A., and Lee, T. (2015). Conservative phase-field lattice Boltzmann model for interface tracking equation. Phys Rev E Stat Nonlin Soft Matter Phys 91(6):063309. DOI: 10.1103/PhysRevE.91.063309.
- 12. Wang, H.L., Chai, Z.H., Shi, B.C., and Liang, H. (2016). Comparative study of the lattice Boltzmann models for Allen-Cahn and Cahn-Hilliard equations. Phys Rev E 94(3-1):033304. DOI: 10.1103/PhysRevE.94.033304.
- 13. Sun, Y., and Beckermann, C. (2007). Sharp interface tracking using the phase-field equation. Journal of Computational Physics 220(2):626-653. DOI: 10.1016/j.jcp.2006.05.025.
- 14. Chiu, P.-H., and Lin, Y.-T. (2011). A conservative phase field method for solving incompressible two-phase flows. Journal of Computational Physics 230(1):185-204. DOI: 10.1016/j.jcp.2010.09.021.
- 15. Liang, H., Wang, R., Wei, Y., and Xu, J. (2023). Lattice Boltzmann method for interface capturing. Phys Rev E 107(2-2):025302. DOI: 10.1103/PhysRevE.107.025302.


- 16. Guo, Z., and Zheng, C. (2008). Analysis of lattice Boltzmann equation for microscale gas flows: Relaxation times, boundary conditions and the Knudsen layer. International Journal of Computational Fluid Dynamics 22(7):465-473. DOI: 10.1080/10618560802253100.
- 17. An, S., Yu, H., Wang, Z., et al. (2017). Unified mesoscopic modeling and GPU-accelerated computational method for image-based pore-scale porous media flows. International Journal of Heat and Mass Transfer 115:1192-1202. DOI: 10.1016/j.ijheatmasstransfer.2017.08.099.
- 18. He, X., Chen, S., and Zhang, R. (1999). A lattice Boltzmann scheme for incompressible multiphase flow and its application in simulation of Rayleigh–Taylor instability. Journal of computational physics 152(2):642-663. DOI.
- 19. Zhang, C., Wang, L.-P., Liang, H., and Guo, Z. (2023). Central-moment discrete unified gas-kinetic scheme for incompressible two-phase flows with large density ratio. Journal of Computational Physics 482. DOI: 10.1016/j.jcp.2023.112040.
- 20. Sitompul, Y.P., and Aoki, T. (2019). A filtered cumulant lattice Boltzmann method for violent two-phase flows. Journal of Computational Physics 390:93-120. DOI: 10.1016/j.jcp.2019.04.019.
- 21. Fakhari, A., and Bolster, D. (2017). Diffuse interface modeling of three-phase contact line dynamics on curved boundaries: A lattice Boltzmann model for large density and viscosity ratios. Journal of Computational Physics 334:620-638. DOI: 10.1016/j.jcp.2017.01.025.
- 22. Ding, H., and Spelt, P.D. (2007). Wetting condition in diffuse interface simulations of contact line motion. Phys Rev E Stat Nonlin Soft Matter Phys 75(4 Pt 2):046708. DOI: 10.1103/PhysRevE.75.046708.
- 23. Wang, L., Huang, H.B., and Lu, X.Y. (2013). Scheme for contact angle and its hysteresis in a multiphase lattice Boltzmann method. Phys Rev E Stat Nonlin Soft Matter Phys 87(1):013301. DOI: 10.1103/PhysRevE.87.013301.
- 24. Zhang, S., Tang, J., and Wu, H. (2022). Wetting boundary schemes in modified phase-field lattice Boltzmann method for binary fluids with large density ratios. Computers & Mathematics with Applications 113:243-253. DOI: 10.1016/j.camwa.2022.03.023.
- 25. Li, Q., Yu, Y., and Luo, K.H. (2019). Implementation of contact angles in pseudopotential lattice Boltzmann simulations with curved boundaries. Physical Review E 100(5-1):053313. DOI: 10.1103/PhysRevE.100.053313.
- 26. Zarareh, A., Khajepor, S., Burnside, S.B., and Chen, B. (2021). Improving the staircase approximation for wettability implementation of phase-field model: Part 1 – Static contact angle. Computers & Mathematics with Applications 98:218-238. DOI: 10.1016/j.camwa.2021.07.013.
- 27. Yang, J., Dai, X., Xu, Q., et al. (2022). Comparative investigation of a lattice Boltzmann boundary treatment of multiphase mass transport with heterogeneous chemical reactions. Physical Review E 105(5):055302. DOI: 10.1103/PhysRevE.105.055302.
- 28. Di Palma, P.R., Huber, C., and Viotti, P. (2015). A new lattice Boltzmann model for interface reactions between immiscible fluids. Advances in Water Resources 82:139-149. DOI: 10.1016/j.advwatres.2015.05.001.
- 29. Chen, L., Kang, Q., Tang, Q., et al. (2015). Pore-scale simulation of multicomponent multiphase reactive transport with dissolution and precipitation. International Journal of Heat and Mass Transfer 85:935-949. DOI: 10.1016/j.ijheatmasstransfer.2015.02.035.


- 30. Li, L., Chen, C., Mei, R., and Klausner, J.F. (2014). Conjugate heat and mass transfer in the lattice Boltzmann equation method. Physical Review E 89(4):043308. DOI: 10.1103/PhysRevE.89.043308.
- 31. Riaud, A., Zhao, S., Wang, K., et al. (2014). Lattice-Boltzmann method for the simulation of multiphase mass transfer and reaction of dilute species. Physical Review E 89(5):053308. DOI: 10.1103/PhysRevE.89.053308.
- 32. Yang, J., Dai, X., Xu, Q., et al. (2021). Lattice Boltzmann modeling of interfacial mass transfer in a multiphase system. Physical Review E 104(1):015307. DOI: 10.1103/PhysRevE.104.015307.
- 33. Zhao, S., Riaud, A., Luo, G., et al. (2015). Simulation of liquid mixing inside micro-droplets by a lattice Boltzmann method. Chemical Engineering Science 131:118-128. DOI: 10.1016/j.ces.2015.03.066.
- 34. Tan, Z., Yan, H., Huang, R., et al. (2022). Phase-field lattice Boltzmann method for the simulation of gas-liquid mass transfer. Chemical Engineering Science 253. DOI: 10.1016/j.ces.2022.117539.
- 35. Mo, H., Yong, Y., Yu, K., et al. (2023). An integrated Lattice-Boltzmann model of immiscible two-phase flow and bulk mass transfer with Marangoni effect. Journal of Computational Physics 481. DOI: 10.1016/j.jcp.2023.112037.
- 36. Haroun, Y., Legendre, D., and Raynal, L. (2010). Volume of fluid method for interfacial reactive mass transfer: Application to stable liquid film. Chemical Engineering Science 65(10):2896-2909. DOI: 10.1016/j.ces.2010.01.012.
- 37. Chen, L., Zhang, R., Kang, Q., and Tao, W.Q. (2020). Pore-scale study of pore-ionomer interfacial reactive transport processes in proton exchange membrane fuel cell catalyst layer. Chemical Engineering Journal 391:123590. DOI: 10.1016/j.cej.2019.123590.
- 38. He, Y.L., Liu, Q., Li, Q., and Tao, W.Q. (2019). Lattice Boltzmann methods for single-phase and solid-liquid phase-change heat transfer in porous media: A review. International Journal of Heat and Mass Transfer 129:160-197. DOI: 10.1016/j.ijheatmasstransfer.2018.08.135.
- 39. Li, Q., Luo, K.H., Kang, Q.J., et al. (2016). Lattice Boltzmann methods for multiphase flow and phase-change heat transfer. Progress in Energy and Combustion Science 52:62-105. DOI: 10.1016/j.pecs.2015.10.001.
- 40. Kang, Q., Lichtner, P.C., and Zhang, D. (2006). Lattice Boltzmann pore-scale model for multicomponent reactive transport in porous media. Journal of Geophysical Research: Solid Earth 111(B5):B05203. DOI: 10.1029/2005jb003951.
- 41. Kou, X., Feng, J.-C., Li, X.-S., et al. (2022). Visualization of interactions between depressurization-induced hydrate decomposition and heat/mass transfer. Energy 239. DOI: 10.1016/j.energy.2021.122230.
- 42. Chaouachi, M., Falenty, A., Sell, K., et al. (2015). Microstructural evolution of gas hydrates in sedimentary matrices observed with synchrotron X-ray computed tomographic microscopy. Geochemistry, Geophysics, Geosystems 16(6):1711-1722. DOI: 10.1002/2015gc005811.
- 43. Yu, P.-Y., Sean, W.-Y., Yeh, R.-Y., et al. (2017). Direct numerical simulation of methane hydrate dissociation in pore-scale flow by using CFD method. International Journal of Heat and Mass Transfer 113:176-183. DOI: 10.1016/j.ijheatmasstransfer.2017.05.053.
- 44. Wang, X., Dong, B., Li, W., et al. (2018). Microscale effects on methane hydrate dissociation at low temperature in the micro porous media channels by depressurization. International


- Journal of Heat and Mass Transfer 122:1182-1197. DOI: 10.1016/j.ijheatmasstransfer.2018.02.056.
- 45. Zhang, L., Zhang, C., Zhang, K., et al. (2019). Pore-Scale Investigation of Methane Hydrate Dissociation Using the Lattice Boltzmann Method. Water Resources Research 55(11):8422-8444. DOI: 10.1029/2019wr025195.
- 46. Zhang, Y., Wang, X., Dong, B., et al. (2023). Numerical simulation of methane hydrate dissociation characteristics in microporous media using lattice Boltzmann method: Effect of fluid flow. Chemical Engineering Science 267. DOI: 10.1016/j.ces.2022.118384.
- 47. Chen, X., Espinoza, D.N., Luo, J.S., et al. (2020). Pore-scale evidence of ion exclusion during methane hydrate growth and evolution of hydrate pore-habit in sandy sediments. Marine and Petroleum Geology 117:104340. DOI: 10.1016/j.marpetgeo.2020.104340.
- 48. Yang, J., Xu, Q., Liu, Z., et al. (2023). Upscaling methane hydrate dissociation kinetic model during depressurisation. Chemical Engineering Science 275. DOI: 10.1016/j.ces.2023.118742.
- 49. Moridis, G.J. (2012). TOUGH+ HYDRATE v1. 2 User's manual: a code for the simulation of system behavior in hydrate-bearing geologic media.
- 50. Rahimi, A., Kasaeipoor, A., Hasani Malekshah, E., et al. (2019). Lattice Boltzmann simulation of 3D natural convection in a cuboid filled with KKL-model predicted nanofluid using Dual-MRT model. International Journal of Numerical Methods for Heat & Fluid Flow 29(1):365-387. DOI: 10.1108/hff-07-2017-0262.
- 51. Wang, G., Yang, J., Lei, T., et al. (2023). A three-dimensional non-orthogonal multiple-relaxation-time phase-field lattice Boltzmann model for multiphase flows at large density ratios and high Reynolds numbers. International Journal of Multiphase Flow 168. DOI: 10.1016/j.ijmultiphaseflow.2023.104582.


