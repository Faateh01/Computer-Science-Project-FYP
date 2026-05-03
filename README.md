# Computer-Science-Project-FYP

What is LungAI?
LungAI is a web-based AI system that predicts whether a patient is at HIGH RISK or LOW RISK of lung cancer based on 15 health and lifestyle features they enter through a form. It runs two machine learning models simultaneously and shows both results side by side with confidence percentages.

The 4 Pages
1. Home Page (/)
The landing page. Shows the project title, key stats (96% LR accuracy, 89% DT accuracy, 15 features, 2 models), and six feature cards explaining what the system does. Has a "Start Prediction" button that takes you to the form.
2. Predict Page (/predict/)
The main page. Has a form split into three sections:

Patient Demographics — Age and Gender
Lifestyle — Smoking, Alcohol, Peer Pressure, Anxiety
Symptoms — Yellow Fingers, Chronic Disease, Fatigue, Wheezing, Coughing, Shortness of Breath, Chest Pain, Swallowing Difficulty

When you click Run AI Prediction, JavaScript sends the data to the server without refreshing the page. The results panel on the right instantly shows:

Logistic Regression → label (HIGH/LOW RISK) + probability bar
Decision Tree → label + probability bar
Overall consensus verdict (HIGH RISK if either model flags it)
An ethical disclaimer reminding users it's not a medical diagnosis

3. Results Page (/results/)
Shows the full performance evaluation of both models with metric bars, numbers, and colour-coded confusion matrices so you can see exactly how reliable each model is.
4. About Page (/about/)
Explains what the project does, how it was built step by step, the tech stack used, and project info (module, dataset, records, features).

How a Prediction Works (Behind the Scenes)

User fills the form and clicks the button
JavaScript collects all 15 values and sends them as JSON to /api/predict/
Django receives the data and puts it into the correct feature order
The input is scaled and fed into Logistic Regression → returns label + probability
The unscaled input is fed into Decision Tree → returns label + probability
If either model says HIGH RISK, the consensus is HIGH RISK
All results are sent back as JSON and the page updates instantly — no reload
