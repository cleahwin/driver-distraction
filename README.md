# Detecting Driver Distraction
Approach for detecting driver distraction
## Abstract
Distracted driving leads to 1,000 deaths daily in the US alone. Although current automobile safety systems exist, they do not incorporate a measure of distraction itself. The purpose of our project was to develop a machine learning-based model that decodes neural data into a measure of distraction, which could be incorporated into future safety systems. To train our system, we collected electroencephalography (EEG) data using a wireless Muse headband while participants played City Car Driving. In every trial of driving, each participant was presented either with no distraction or a randomly chosen distraction. We used data processing techniques, to normalize and clean our data, and tested supervised machine learning models to decode relative frequency band powers into distraction. For binary classification to detect whether an individual was distracted, we tested a logistic regression model (71.3% accuracy) and a multilayer perceptron (81.2% accuracy). Next, we designed a distraction metric on a scale of 0 to 5. We implemented a multivariable linear regression (mse = 0.93), and a decision tree classifier (71% accuracy) to predict this metric. The multilayer perceptron most effectively detected the presence of distraction, and the decision tree best predicted distraction level. Using recursive feature elimination, we determined that the delta and beta frequency bands drove our classifier performance. With this research, we demonstrated that a commercially-available wireless EEG system can decode neural data into distraction. Our EEG-based early driver distraction detection system has the potential to be incorporated into safety systems to improve driver safet

---
## Project Details
### Rationale
On January 6th 2020, three people were killed and fourteen were injured when a trucker in Indiana crashed into eight cars. Surprisingly, this entire accident was caused by a single person distracted by his coffee mug. The consequences of distracted driving can be disastrous. According to the CDC, distracted driving accidents kill approximately 9 people and injure 1,000 people daily in the USA. Our project aims to detect distracted driving using EEG data and prevent potential accidents. In the future, our system can be integrated with current safety systems in vehicles to predict accidents earlier and take appropriate actions. Our project has the potential to reduce automobile accidents and save thousands of lives. 

### Research Goal
Our research question explores how distracted driving manifests in data from EEG brain sensors. We hypothesize that there is a correlation between neural activity and distracted driving. Our goal is to thereby create a machine learning model that can learn this correlation using EEG neural activity data from a Muse headband. Finally, we propose a method for EEG data used to detect distracted driving to be incorporated into existing driver safety systems.

### Procedure
For the purposes of our experiment, we recruited participants, who were adult drivers. We did not target any particular ethinic group, race, or population. Participants were recruited by asking members in our family, our neighbors, and students at the University of Washington. The total time for each experiment was 45 minutes. First, we asked our participants to read and fill the informed consent form. Participation in the experiment was completely voluntary and participants were given the right to stop/withdraw at any time. Next, the participants were instructed how to successfully fit the Muse 2 headband, which is used to get EEG data, to ensure good signal quality. The participants were given around 5 minutes to practice and familiarize themselves with the driving simulator. After this, we started recording the EEG data using an app called MuseDirect. We also took a screen recording of the driving simulator. We recorded timepoints of when the distraction was started and ended. Three driving tasks were assigned to each subject (drive to the shopping mall, drive to service station, or driving to a parking lot - all on the simulator). For each task, three distractions (randomly chosen from the list below) were presented to the participants at different times during the task. Here is a list of possible distractions: 
  - Solving 5 math problems we ask (e.g., 34+57) 
  - Write a list of fruits on their phone/text someone 
  - Take a sip of water
  - Pick up a ball dropped next to the driver’s seat
  - Answer a riddle
  - Conversation about favorite food
  - Play I spy (e.g., I spy a red building)
  - Say a tongue twister

After the driving tasks have been completed, we stopped the EEG recording and allowed the participants to remove the Muse Headband. We asked the participants to rate how distracted they felt during the distracting task on a scale from 1 (least distracted) to 5 (most distracted). Finally, we debriefed and thanked the participants, answering any questions they had. All data was kept anonymous and confidential.
The flowchart below is a brief summary of the procedure described above. 

![image](https://user-images.githubusercontent.com/20733329/147866189-fbda92a6-c08e-4c4e-a105-34a50ddac6ad.png)

### Data Analysis
We used an app called MuseDirect, which allows us to download EEG data to a CSV file containing the EEG data of a person wearing a MuseHeadband. After an experiment is completed, we analyzed the screen recording and extracted speed and road curvature data from the videos. We created a binary classifier which predicted whether the participant was distracted. This program analyzed data (for this classifier, we used alpha, beta, theta, gamma, and delta waves) to determine if the distracted value was a 0 (not distracted) or 1 (distracted). We also tested other machine learning models for optimal accuracy. Python and Sci-kit libraries were used for the models in this project.

![image](https://user-images.githubusercontent.com/20733329/147865337-9a87211c-5a60-4606-b990-9402748cb6e2.png)

### Driver Distraction Models
We created two models to determine whether a driver was distracted. We used the collected EEG data to do this. Here are the two models:
  - We created a logistic regression model which learned a logistic function to determine whether the driver is distracted or not at a certain time (1 or 0). The accuracy of this model was 79%. Below is the confusion matrix of this model.
  - 
![image](https://user-images.githubusercontent.com/20733329/147865857-73edcf61-e091-471c-ad7c-5f8c28c25db1.png)
  - We created a multilayer perceptron to predict whether the driver was distracter. The accuracy of the model was 82%, and below is the confusion matrix.
 
![image](https://user-images.githubusercontent.com/20733329/147865923-627a0535-93a2-488f-a678-5dc2af8f0e2e.png)

We created two models to determine how distracted the driver was. We created a metric (shown in the table below) based on a subject score of the visibility of distraction in the participants' driving. We found this value to be correlated with the number of swerves. The R-value was 0.74. 

![image](https://user-images.githubusercontent.com/20733329/147866088-1ef729fa-f7c1-465c-a250-7ab6efb344a1.png)

Below is a short description of the two models:
  - We created a linear regression model learns a linear model to predict a level of distraction from 1-5. The mean squared error was 1.34. This graph shows predicted (orange) versus actual (blue) distraction values for 25 samples from the testing data. Bars highlighted green have roughly no error.

![image](https://user-images.githubusercontent.com/20733329/147866112-0df30776-05aa-4dff-a5df-926b8412ecb1.png)
  - We created a decision tree which learns a tree that is used to predict distraction level from 1-5. The accuracy was 72%. The confusion matrix for this model is below.

![image](https://user-images.githubusercontent.com/20733329/147866164-9a3ee92e-9c4f-43fe-a94a-fa93f830b679.png)

### Recursive Feature Elimination
Recursive feature elimination was used to determine importance of each feature in the logistic regression model. The coefficients of a logistic regression model for binary classification, where the twenty feature are the five waves for each of the four electrodes. The values in yellow below are the most important features.

![image](https://user-images.githubusercontent.com/20733329/147866259-d2378853-b10a-4a9f-b72d-4b56943d833b.png)

After performing recursive feature elimination, we concluded that the delta and beta values maximize driver classifier performance. Beta values are associated with cognitive tasks, while delta waves are slower, lower frequency waves. This supports our results, because they may account for differing amounts of focus required when driving with or without distraction.

### Conclusion
For binary classification, the multilayer perceptron has a higher accuracy than the logistic regression model. This may be because our neural network has multiple layers and can be more powerful for complex data, especially when the data from different classes is not linearly separable. For predicting distraction level, the decision tree model performed better than the linear regression model. This could mean that the features are not linearly related to the distraction level. The decision tree, on the other hand, could capture non-linear relationships in the data. This model worked better with a higher tree depth, which is because a higher depth makes the model more powerful.

### Future Work
There are multiple steps we would like to take to improve the accuracy of our model:
  - Collect more experimental data to train our models
  - Revise the distraction level value so it better captures distraction level
  - Perform experiment on a real driving simulator (with a steering wheel, brakes…)
  - Experiment using other algorithms and data processing techniques including Ridge Regression and Convolution Neural Network
  - Collect more data on car stats during the experiment
We plan to implement this end to end system in cars with the current safety systems. The diagram below is how we envision our system integrating with current car systems.

![image](https://user-images.githubusercontent.com/20733329/147866375-2caf585a-6175-4d76-bbf4-98dca51bf863.png)

---
## Acknowledgments
We are deeply grateful to Courtnie Paschall (MD/PhD Student at the University of Washington) for her guidance. Our thanks to Nikolas Ioannou, a UW student, for his inputs. We are also very thankful to our family who supported us over the course of this project. Lastly, we thank the participants who kindly volunteered to help us collect our data.

---
## Files
Here is a brief description of the main files you will find in this repository:
- ``project_poster.png `` is the poster we presented at the 2020 *Washington State Science & Engineering Fair* (WSSEF). It won first place at this fair.
