# Analyzing Mars atmosphere with machine learning
The task was to create simple algorithms to predict various quantities in the Martian atmosphere, that have the potential to be implemented in the existing NASA Mars general circulation model (written in fortran). These algorithms should be computationally significantly faster than their respective fortran code snippets, leading to considerably faster simulation times.
Due to the large number of features and nonlinearities in the data, the algorithms were motivated by machine learning algorithms.

## Bottom Solar Flux

The first quantity that was supposed to be predicted was bottom solar flux. The final copy of the jupyter notebook for this algorithm is **ML3** (all of the ML2... notebooks were rough work to understand how to construct the predictor). 

The features used to predict the bottom solar flux (GSW) were temperature profile (obtained in box 3 of the notebook), dust opacity (TAU_OD, box 9), surface temperature (TSK, box 7), top solar flux (TOASW, box 11), and albedo (box 6, ALBEDO).
The only contribution of albedo is that the bottom solar flux data contains a (1-albedo) factor, so the value that is actually being predicted in this notebook is bsf2 = bsf/(1-albedo), seen in box 15.

Mars has a pretty tenuous atmosphere, which makes the bottom solar flux have a strong, approximately linear dependence on the top solar flux. In other words, the intensity of light at the surface of Mars is very dependent on the intensity of light coming towards Mars at the top of the atmosphere.
Thus, a linear regressor-based approach seemed appropriate. Another benefit of linear regressors is that they are very easy to implement into fortran code compared to other machine learning algorithms.

#### Piecewise Linear Regressors

Unfortunately, just using a linear regressor to fit the features with the bottom solar flux was not good enough. The nonlinear effects were not neglible, specifically in the extrema of the bottom solar flux range, where the effects of the other features (besides the top solar flux) become more prominent.
To remedy the underpredictions caused by nonlinearities, a piecewise function was created, composed of n linear regressors (n is some natural number), where each linear regressor operates for a certain subrange of top solar flux values. 
The intuition behind this was that each regressor only has to deal with a small interval of the whole feature space, allowing a better fit as the interval gets smaller (like how a secant line interval gets better at interpolating a curve as its length approaches 0).

The different intervals were created by splitting the top solar flux range into n parts, where the **i**th part is the top solar flux being in the (i-1)×(100/n) to (i)×(100/n) percentiles. In the notebook, n = 10. The regressors are generated in the function sat() in box 21 and stored as a list. 
To predict the bottom solar flux for a given data point of input values, the special prediction function s_predict() in box 23 looks at which two percentile bounds the top solar flux falls into. If it lies in between the (i-1)×(100/n) to (i)×(100/n) percentiles, the **i**th linear regressor from the linear regressor list is used to predict the bottom solar flux.

#### Relative Error Minimizing

Since these regressors minimize the mean square error, a prediction for the bottom solar flux of 103 W/m^2 where the actual is 100 W/m^2 is equivalent to a prediction of 4 W/m^2 where the actual is 1 W/m^2. 
This is problematic from a physical standpoint, since the relative error (the difference between predicted and real divided by real) in the first case is 3%, and in the second case is 300%!
To minimize these massive relative errors occuring at the low bottom solar flux locations, weights were assigned to each point in the training data, equal to 1/(bottom solar flux)^2.
This makes each summand term in the mean square error summation equal to ((prediction - real)/(real))^2 instead of the previous (prediction - real)^2, so now the relative error (prediction - real)/(real) is being minimized. The function that outputs this relative error minimizing linear regressor list is satr(), in box 22 of the notebook. 
Although this does lead to significantly smaller relative errors at the low flux regions, it leads to higher absolute error. 
When the errors are integrated across the whole atmsophere, the average flux error contribution for the relative error linear regressors obtained from satr() is noticeably bigger than the average flux error contribution for the regular linear regressors obtained from sat().
This is why the regular piecewise linear regressors obtained from sat() are better in general to implement into the fortran code, unless more accuracy is required near the very low flux regions.

#### Results

The linear regressors are trained on the first ten time snapshots of the wrfout_d01_0002-00232_06\:00\:00 file, located in the /export/data/lee/1-Projects/WRF_Radiation/run/ directory on the rainy server.
After predicting the bottom solar flux (with both sat() and satr() approaches), the root mean square error is calculated. 
Then, the errors in each cell are integrated together for each snapshot, then divided by the surface area of mars, to obtain the "additional" average flux added to each latitude-longtitude cell (in boxes 28, 29, 40) for each of the 10 time snapshots.
Finally, several latitude-longtitude maps are shown that display how accurate the prediction was in different places. 
**All of these values displayed, except for the relative errors, are in units of W/m^2. This is the same for all of the further results described below.**

## Down Infrared Flux

The features used to predict the downward infrared flux were temperature, albedo, dust profile, surface temperature, and top solar flux. The final copy of the prediction is **ML3.1.2**.
Since the downward IR flux is dependent on top solar flux in a very similar manner to the bottom solar flux, the predictor is essentially the same as in ML3. 
Once again, n=10 linear regressors are trained on certain subsections of the training data corresponding to certain ranges of the top solar flux. 
The ith regressor in the regressor list is used when the top solar flux value in the input lies between the (i-1)×(100/n) to (i)×(100/n) percentiles. 
One thing to keep in mind is that 53 downward flux values are predicted for each input data point no instead of just 1 bottom solar flux, since downward infrared flux has a height dimension (that contains 53 lattice points).

#### Results

The linear regressors are trained on the first ten time snapshots of the wrfout_d01_0002-00172_06\:00\:00 file (approximately corresponding to summer on Mars), located in the /export/data/lee/1-Projects/WRF_Radiation/run/ directory on the rainy server.
After predicting the bottom solar flux, the root mean square error is calculated. 
Then, the errors in each cell are integrated together and summed across all 53 height layers, then divided by the surface area of mars and number of latitude-longtitude layers, to obtain the "additional" average flux added to each latitude-longtitude cell for each of the ten time snapshots (in boxes 22, 23).
Finally, several latitude-longtitude maps and height-longtitude maps for the equator are shown that display how accurate the prediction was in different places. 

## Upward Infrared Flux

The last quantity that was attempted to be predicted was upward infrared flux. A very approximate calculation of the upward infrared flux is:
emiss×TSK^4 × exp(-opacity_from_surface) + integral(temp^4 × exp(-opacity_from_layer) dh), where the integral is calculated vertically,  dh is an infitesimal height element, and it starts on the surface of Mars.
Since I only have 52 height layers to use, I had to do a discretized version of this integral, which meant a linear combination of the emiss×TSK^4 and all the temp^4 × exp(-opacity_from_layer) terms.
This motivated a linear regressor approach where the emiss×TSK^4 and all the temp^4 × exp(-opacity_from_layer) terms were the features, and the regressor figures out the optimal coefficients.
Unfortunately, there were still considerable errors, so Chris suggested adding in features in the form of exp(-pressure) × temp^4 × ΔP, where ΔP is the pressure difference between consecutive layers. These should approximately describe blackbody radiation coming from each layer according to the Stefan Boltzmann equation. An issue with this is that the first 20 features (corresponding to the bottom 20 layers) are just 0 (box 4 of the **ML3.3.8** notebook), since the pressure is in the hundreds of its respective unit, causing exp(-pressure) to be extremely small. This renders these first 20 features useless in predicting the upward infrared flux.

#### Results

The linear regressor was trained on data from all four seasons, and the final version of the code is **ML3.3.8**. 

The af_summer, wf_summer files are auxhist11_d01_0002-00172_06\:00\:00, wrfout_d01_0002-00172_06\:00\:00.
The af_winter, wf_winter files are auxhist11_d01_0002-00532_06\:00\:00, wrfout_d01_0002-00532_06\:00\:00.
The af_fall, wf_fall files are auxhist11_d01_0002-00352_06\:00\:00, wrfout_d01_0002-00352_06\:00\:00.
The af_spring, wf_spring files are auxhist11_d01_0002-00652_06\:00\:00, wrfout_d01_0002-00652_06\:00\:00.
The af_test, wf_test files are auxhist11_d01_0002-00472_06\:00\:00, wrfout_d01_0002-00472_06\:00\:00.

It predicts reasonably well when tested on data that it was trained on (on some seasons better than others), but when new data is tested on (shown in the "Last Test Set" section of the notebook), it does significantly worse, due to a large overprediction of the flux around the South Pole.
For each season, there is a histogram displaying the difference between the prediction and the actual upper infrared flux (for the summer prediction, it is box 19). 
Then there are average error plots which show averaged out versions of the actual, predicted, error, and relative error plots for 8 consecutive time snapshots, so we can gather intuition about where the model predicted the worst.
Then there are standard deviation error plots which show the standard deviation of the actual, predicted and error for 8 consective time snapshots, so the variance in the atmosphere is understood.
Then, the maximum, minimum, standard deviation, and average of the integrated flux errors are calculated for each horizontal layer, to understand the additional flux contribution produced by using the simple algorithm rather than the fortran code.
These results show that spring, summer, and fall are predicted better than winter, but the new data in the "Last Test Set" section prediction was predicted significantly worse. 
We believe that this is due to effects at the South and North Poles.

#### Random Forest Regressor

To try to deal with the nonlinearities, a random forest regressor was trained on the same data (**ML3.3.9**), and had slightly better predictive power (than the linear regressor) when tested on each of the four seasons and new data. The CO2ICE variable is included as well in the input data, so the predictor should consider the level of ice on the Martian surface under the specific point of upper infrared flux it is predicting. This slightly improved the predictions near the South Pole of the "Last Test Set", however they are still far from ideal. 

The individual trees were analyzed to see which features are the most relevant.

Unfortunately, it appears that CO2ICE is the least relevant feature (if the 0-valued features of the exp(-pressure) terms are ignored), as shown in the compute_feature_importances() output in box 166. 
The features used in the first few 20 node for each tree in the regressors are outputted in box 167, and there are some interesting things to notice. Primarily, the 0th feature (the surface temp term) is the most important, and there are some other features like the 23, 19, 20, 29th that appear pretty useful as well.
The total samples went through all of the nodes involving a certain feature was also calculated, shown in boxes 168 to 173. This is prone to overcounting, but essentially the more samples going through a node, the more important it is. These results agree with the compute_feature_importances() calculation.

Further work involving opacities and other features that relate to the upper infrared flux are required to create a better prediction algorithm.

# Supervision & Funding
This summer research project was supervised by Dr. Chris Lee of the University of Toronto, and funded by a NSERC USRA.

