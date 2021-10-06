---


---

<h1 id="get-started-with-artificial-intelligence-on-azure">Get started with artificial intelligence on Azure</h1>
<h2 id="get-started-with-ai-on-azure">Get started with AI on Azure</h2>
<h3 id="introduction-to-ai">Introduction to AI</h3>
<h4 id="what-is-ai">What is AI?</h4>
<p>Simply put, AI is the creation of software that imitates human behaviors and capabilities. Key elements include:</p>
<ul>
<li><strong>Machine learning</strong>  - This is often the foundation for an AI system, and is the way we “teach” a computer model to make prediction and draw conclusions from data.</li>
<li><strong>Anomaly detection</strong>  - The capability to automatically detect errors or unusual activity in a system.</li>
<li><strong>Computer vision</strong>  - The capability of software to interpret the world visually through cameras, video, and images.</li>
<li><strong>Natural language processing</strong>  - The capability for a computer to interpret written or spoken language, and respond in kind.</li>
<li><strong>Conversational AI</strong>  - The capability of a software “agent” to participate in a conversation.</li>
</ul>
<h3 id="understand-machine-learning">Understand machine learning</h3>
<p>Machine Learning is the foundation for most AI solutions.</p>
<p>Machines learn from data. Data scientists can use all of that data to train machine learning models that can make predictions and inferences based on the relationships they find in the data.</p>
<p>For example, suppose an environmental conservation organization wants volunteers to identify and catalog different species of wildflower using a phone app.</p>
<ol>
<li>A team of botanists and scientists collect data on wildflower samples.</li>
<li>The team labels the samples with the correct species.</li>
<li>The labeled data is processed using an algorithm that finds relationships between the features of the samples and the labeled species.</li>
<li>The results of the algorithm are encapsulated in a model.</li>
<li>When new samples are found by volunteers, the model can identify the correct species label.</li>
<li></li>
</ol>
<h4 id="machine-learning-in-microsoft-azure">Machine learning in Microsoft Azure</h4>
<p>Microsoft Azure provides the  <strong>Azure Machine Learning</strong>  service - a cloud-based platform for creating, managing, and publishing machine learning models. Azure Machine Learning provides the following features and capabilities:</p>
<p>MACHINE LEARNING IN MICROSOFT AZURE</p>
<p>Feature</p>
<p>Capability</p>
<p>Automated machine learning</p>
<p>This feature enables non-experts to quickly create an effective machine learning model from data.</p>
<p>Azure Machine Learning designer</p>
<p>A graphical interface enabling no-code development of machine learning solutions.</p>
<p>Data and compute management</p>
<p>Cloud-based data storage and compute resources that professional data scientists can use to run data experiment code at scale.</p>
<p>Pipelines</p>
<p>Data scientists, software engineers, and IT operations professionals can define pipelines to orchestrate model training, deployment, and management tasks.</p>
<h3 id="understand-anomaly-detection">Understand anomaly detection</h3>
<p>-An anomaly detection model is trained to understand expected fluctuations in the telemetry measurements over time.<br>
In Microsoft Azure, the <strong>Anomaly Detector</strong> service provides an application programming interface (API) that developers can use to create anomaly detection solutions.</p>
<h3 id="understand-computer-vision">Understand computer vision</h3>
<p>Computer Vision is an area of AI that deals with visual processing.</p>
<h2 id="computer-vision-models-and-capabilities">Computer Vision models and capabilities</h2>
<p>Most computer vision solutions are based on machine learning models that can be applied to visual input from cameras, videos, or images. The following table describes common computer vision tasks.</p>
<p>Computer Vision models and capabilities</p>
<p>Task</p>
<p>Description</p>
<p>Image classification</p>
<p><img src="https://docs.microsoft.com/en-us/learn/wwl-data-ai/get-started-ai-fundamentals/media/image-classification.png" alt="An image of a taxi with the label &quot;Taxi&quot;"><br>
Image classification involves training a machine learning model to classify images based on their contents. For example, in a traffic monitoring solution you might use an image classification model to classify images based on the type of vehicle they contain, such as taxis, buses, cyclists, and so on.</p>
<p>Object detection</p>
<p><img src="https://docs.microsoft.com/en-us/learn/wwl-data-ai/get-started-ai-fundamentals/media/object-detection.png" alt="An image of a street with buses, cars, and cyclists identified and highlighted with a bounding box"><br>
Object detection machine learning models are trained to classify individual objects within an image, and identify their location with a bounding box. For example, a traffic monitoring solution might use object detection to identify the location of different classes of vehicle.</p>
<p>Semantic segmentation</p>
<p><img src="https://docs.microsoft.com/en-us/learn/wwl-data-ai/get-started-ai-fundamentals/media/semantic-segmentation.png" alt="An image of a street with the pixels belonging to buses, cars, and cyclists identified"><br>
Semantic segmentation is an advanced machine learning technique in which individual pixels in the image are classified according to the object to which they belong. For example, a traffic monitoring solution might overlay traffic images with “mask” layers to highlight different vehicles using specific colors.</p>
<p>Image analysis</p>
<p><img src="https://docs.microsoft.com/en-us/learn/wwl-data-ai/get-started-ai-fundamentals/media/image-analysis.png" alt="An image of a person with a dog on a street and the caption &quot;A person with a dog on a street&quot;"><br>
You can create solutions that combine machine learning models with advanced image analysis techniques to extract information from images, including “tags” that could help catalog the image or even descriptive captions that summarize the scene shown in the image.</p>
<p>Face detection, analysis, and recognition</p>
<p><img src="https://docs.microsoft.com/en-us/learn/wwl-data-ai/get-started-ai-fundamentals/media/face-analysis.png" alt="An image of multiple people on a city street with their faces highlighted"><br>
Face detection is a specialized form of object detection that locates human faces in an image. This can be combined with classification and facial geometry analysis techniques to infer details such as age and emotional state; and even recognize individuals based on their facial features.</p>
<p>Optical character recognition (OCR)</p>
<p><img src="https://docs.microsoft.com/en-us/learn/wwl-data-ai/get-started-ai-fundamentals/media/ocr.png" alt="An image of a building with the sign &quot;Toronto Dominion Bank&quot;, which is highlighted"><br>
Optical character recognition is a technique used to detect and read text in images. You can use OCR to read text in photographs (for example, road signs or store fronts) or to extract information from scanned documents such as letters, invoices, or forms.</p>
<h2 id="computer-vision-services-in-microsoft-azure">Computer vision services in Microsoft Azure</h2>

<table>
<thead>
<tr>
<th>Service</th>
<th>Capabilities</th>
</tr>
</thead>
<tbody>
<tr>
<td><strong>Computer Vision</strong></td>
<td>You can use this service to analyze images and video, and extract descriptions, tags, objects, and text.</td>
</tr>
<tr>
<td><strong>Custom Vision</strong></td>
<td>Use this service to train custom image classification and object detection models using your own images.</td>
</tr>
<tr>
<td><strong>Face</strong></td>
<td>The Face service enables you to build face detection and facial recognition solutions.</td>
</tr>
<tr>
<td><strong>Form Recognizer</strong></td>
<td>Use this service to extract information from scanned forms and invoices.</td>
</tr>
</tbody>
</table><h3 id="understand-natural-language-processing">Understand natural language processing</h3>
<p>Natural language processing (NLP) is the area of AI that deals with creating software that understands written and spoken language.</p>
<p>NLP enables you to create software that can:</p>
<ul>
<li>Analyze and interpret text in documents, email messages, and other sources.</li>
<li>Interpret spoken language, and synthesize speech responses.</li>
<li>Automatically translate spoken or written phrases between languages.</li>
<li>Interpret commands and determine appropriate actions.</li>
</ul>
<h4 id="natural-language-processing-in-microsoft-azure">Natural language processing in Microsoft Azure</h4>

<table>
<thead>
<tr>
<th>Service</th>
<th>Capabilities</th>
</tr>
</thead>
<tbody>
<tr>
<td><strong>Text Analytics</strong></td>
<td>Use this service to analyze text documents and extract key phrases, detect entities (such as places, dates, and people), and evaluate sentiment (how positive or negative a document is).</td>
</tr>
<tr>
<td><strong>Translator Text</strong></td>
<td>Use this service to translate text between more than 60 languages.</td>
</tr>
<tr>
<td><strong>Speech</strong></td>
<td>Use this service to recognize and synthesize speech, and to translate spoken languages.</td>
</tr>
<tr>
<td><strong>Language Understanding Intelligent Service (LUIS</strong>)</td>
<td>Use this service to train a language model that can understand spoken or text-based commands.</td>
</tr>
</tbody>
</table><h3 id="understand-conversational-ai">Understand conversational AI</h3>
<p>Conversational AI is the term used to describe solutions where AI agents participate in conversations with humans. Most commonly, conversational AI solutions use <em>bots</em> to manage dialogs with users. These dialogs can take place through web site interfaces, email, social media platforms, messaging systems, phone calls, and other channels.</p>
<h4 id="conversational-ai-in-microsoft-azure">Conversational AI in Microsoft Azure</h4>

<table>
<thead>
<tr>
<th>Service</th>
<th>Capabilities</th>
</tr>
</thead>
<tbody>
<tr>
<td><strong>QnA Maker</strong></td>
<td>This cognitive service enables you to quickly build a  <em>knowledge base</em>  of questions and answers that can form the basis of a dialog between a human and an AI agent.</td>
</tr>
<tr>
<td><strong>Azure Bot Service</strong></td>
<td>This service provides a platform for creating, publishing, and managing bots. Developers can use the  <em>Bot Framework</em>  to create a bot and manage it with Azure Bot Service - integrating back-end services like QnA Maker and LUIS, and connecting to channels for web chat, email, Microsoft Teams, and others.</td>
</tr>
</tbody>
</table><h3 id="challenges-and-risks-with-ai">Challenges and risks with AI</h3>

<table>
<thead>
<tr>
<th>Challenge or Risk</th>
<th>Example</th>
</tr>
</thead>
<tbody>
<tr>
<td>Bias can affect results</td>
<td></td>
</tr>
<tr>
<td>A loan-approval model discriminates by gender due to bias in the data with which it was trained</td>
<td></td>
</tr>
<tr>
<td>Errors may cause harm</td>
<td></td>
</tr>
<tr>
<td>An autonomous vehicle experiences a system failure and causes a collision</td>
<td></td>
</tr>
<tr>
<td>Data could be exposed</td>
<td></td>
</tr>
<tr>
<td>A medical diagnostic bot is trained using sensitive patient data, which is stored insecurely</td>
<td></td>
</tr>
<tr>
<td>Solutions may not work for everyone</td>
<td></td>
</tr>
<tr>
<td>A home automation assistant provides no audio output for visually impaired users</td>
<td></td>
</tr>
<tr>
<td>Users must trust a complex system</td>
<td></td>
</tr>
<tr>
<td>An AI-based financial tool makes investment recommendations - what are they based on?</td>
<td></td>
</tr>
<tr>
<td>Who’s liable for AI-driven decisions?</td>
<td></td>
</tr>
<tr>
<td>An innocent person is convicted of a crime based on evidence from facial recognition – who’s responsible?</td>
<td></td>
</tr>
</tbody>
</table><h3 id="understand-responsible-ai">Understand responsible AI</h3>
<p>At Microsoft, AI software development is guided by a set of six principles</p>
<h4 id="fairness">Fairness</h4>
<p>For example, a machine learning model should make predictions of whether or not the loan should be approved without incorporating any bias based on gender, ethnicity, or other factors that might result in an unfair advantage or disadvantage to specific groups of applicants.</p>
<p>Azure Machine Learning includes the capability to interpret models and quantify the extent to which each feature of the data influences the model’s prediction. This capability helps data scientists and developers identify and mitigate bias in the model.</p>
<h4 id="reliability-and-safety">Reliability and safety</h4>
<p>For example, consider an AI-based software system for an autonomous vehicle; or a machine learning model that diagnoses patient symptoms and recommends prescriptions. Unreliability in these kinds of system can result in substantial risk to human life.</p>
<p>AI-based software application development must be subjected to rigorous testing and deployment management processes to ensure that they work as expected before release.</p>
<h4 id="privacy-and-security">Privacy and security</h4>
<p>The machine learning models on which AI systems are based rely on large volumes of data, which may contain personal details that must be kept private.</p>
<h4 id="inclusiveness">Inclusiveness</h4>
<p>AI should bring benefits to all parts of society, regardless of physical ability, gender, sexual orientation, ethnicity, or other factors.</p>
<h4 id="transparency">Transparency</h4>
<p>AI systems should be understandable. Users should be made fully aware of the purpose of the system, how it works, and what limitations may be expected.</p>
<h4 id="accountability">Accountability</h4>
<p>People should be accountable for AI systems. Designers and developers of AI-based solution should work within a framework of governance and organizational principles that ensure the solution meets ethical and legal standards that are clearly defined.</p>
<h1 id="use-visual-tools-to-create-machine-learning-models-with-azure-machine-learning">Use visual tools to create machine learning models with Azure Machine Learning</h1>
<h2 id="use-automated-machine-learning-in-azure-machine-learning">Use automated machine learning in Azure Machine Learning</h2>
<p><em>Machine Learning</em>  is the foundation for most artificial intelligence solutions, and the creation of an intelligent solution often begins with the use of machine learning to train a predictive model using historic data that you have collected.</p>
<h3 id="what-is-machine-learning">What is machine learning?</h3>
<p>Machine learning is a technique that uses mathematics and statistics to create a model that can predict unknown values.</p>
<p>Mathematically, you can think of machine learning as a way of defining a function (let’s call it  <em><strong>f</strong></em>) that operates on one or more  <em>features</em>  of something (which we’ll call  <em><strong>x</strong></em>) to calculate a predicted  <em>label</em>  (<em><strong>y</strong></em>) - like this:</p>
<p><em><strong>f(x) = y</strong></em></p>
<p>The specific operation that the  <em><strong>f</strong></em>  function performs on  <em>x</em>  to calculate  <em>y</em>  depends on a number of factors, including the type of model you’re trying to create and the specific algorithm used to train the model. Additionally in most cases, the data used to train the machine learning model requires some pre-processing before model training can be performed.</p>
<h3 id="create-an-azure-machine-learning-workspace">Create an Azure Machine Learning workspace</h3>
<p>Azure Machine Learning is a cloud-based platform for building and operating machine learning solutions in Azure. It includes a wide range of features and capabilities that help data scientists prepare data, train models, publish predictive services, and monitor their usage. Most importantly, it helps data scientists increase their efficiency by automating many of the time-consuming tasks associated with training models; and it enables them to use cloud-based compute resources that scale effectively to handle large volumes of data while incurring costs only when actually used.</p>
<h4 id="create-an-azure-machine-learning-workspace-1">Create an Azure Machine Learning workspace</h4>
<ol>
<li>Sign into the <a href="https://portal.azure.com/">Azure portal</a></li>
<li>Select <strong>＋Create a resource</strong>, search for <em>Machine Learning</em>, and create a new <strong>Machine Learning</strong> resource with the following settings:</li>
</ol>
<ul>
<li><strong>Subscription</strong>: <em>Your Azure subscription</em></li>
<li><strong>Resource group</strong>: <em>Create or select a resource group</em></li>
<li><strong>Workspace name</strong>: <em>Enter a unique name for your workspace</em></li>
<li><strong>Region</strong>: <em>Select the geographical region closest to you</em></li>
<li><strong>Storage account</strong>: <em>Note the default new storage account that will be created for your workspace</em></li>
<li><strong>Key vault</strong>: <em>Note the default new key vault that will be created for your workspace</em></li>
<li><strong>Application insights</strong>: <em>Note the default new application insights resource that will be created for your workspace</em></li>
<li><strong>Container registry</strong>: None (<em>one will be created automatically the first time you deploy a model to a container</em>)</li>
</ul>
<h3 id="create-compute-resources">Create compute resources</h3>
<p>At its core, Azure Machine Learning is a platform for training and managing machine learning models, for which you need compute on which to run the training process.</p>
<h4 id="create-compute-targets">Create compute targets</h4>
<p>Compute targets are cloud-based resources on which you can run model training and data exploration processes.</p>
<p>In <a href="https://ml.azure.com/">Azure Machine Learning studio</a>, view the <strong>Compute</strong> page (under <strong>Manage</strong>). There are four kinds of compute resource you can create:</p>
<ul>
<li><strong>Compute Instances</strong>: Development workstations that data scientists can use to work with data and models.</li>
<li><strong>Compute Clusters</strong>: Scalable clusters of virtual machines for on-demand processing of experiment code.</li>
<li><strong>Inference Clusters</strong>: Deployment targets for predictive services that use your trained models.</li>
<li><strong>Attached Compute</strong>: Links to existing Azure compute resources, such as Virtual Machines or Azure Databricks clusters.</li>
</ul>
<h3 id="explore-data">Explore data</h3>
<p>Machine learning models must be trained with existing data.</p>
<h4 id="create-a-dataset">Create a dataset</h4>
<p>In Azure Machine Learning, data for model training and other operations is usually encapsulated in an object called a <em>dataset</em>.</p>
<p>In <a href="https://ml.azure.com/">Azure Machine Learning studio</a>, view the <strong>Datasets</strong> page. You can create a new dataset e.g. <strong>from web files</strong>.</p>
<h3 id="train-a-machine-learning-model">Train a machine learning model</h3>
<p>Azure Machine Learning includes an <em>automated machine learning</em> capability that leverages the scalability of cloud compute to automatically try multiple pre-processing techniques and model-training algorithms in parallel to find the best performing supervised machine learning model for your data.</p>
<p>The automated machine learning capability in Azure Machine Learning supports <em>supervised</em> machine learning models - in other words, models for which the training data includes known label values. You can use automated machine learning to train models for:</p>
<ul>
<li><strong>Classification</strong> (predicting categories or <em>classes</em>)</li>
<li><strong>Regression</strong> (predicting numeric values)</li>
<li><strong>Time series forecasting</strong> (regression with a time-series element, enabling you to predict numeric values at a future point in time)</li>
</ul>
<p>Each possible combination of training algorithm and pre-processing steps is tried and the performance of the resulting model is evaluated.</p>
<p>The best model is identified based on the evaluation metric you specified (e.g. <em>Normalized root mean squared error</em>). To calculate this metric, the training process used some of the data to train the model, and applied a technique called <em>cross-validation</em> to iteratively test the trained model with data it wasn’t trained with and compare the predicted value with the actual known value.</p>
<p>The difference between the predicted and actual value (known as the <em>residuals</em>) indicates the amount of <em>error</em> in the model, and this particular performance metric is calculated by squaring the errors across all of the test cases, finding the mean of these squares, and then taking the square root. What all of this means is that smaller this value is, the more accurately the model is predicting.<br>
Then review the charts, which show the performance of the model by comparing the predicted values against the true values, and by showing the <em>residuals</em> (differences between predicted and actual values) as a histogram.</p>
<p>The <strong>Predicted vs. True</strong> chart should show a diagonal trend in which the predicted value correlates closely to the true value. A dotted line shows how a perfect model should perform, and the closer the line for your model’s average predicted value is to this, the better its performance. A histogram below the line chart shows the distribution of true values.</p>
<p><img src="https://docs.microsoft.com/en-us/learn/wwl-data-ai/use-automated-machine-learning/media/predicted-vs-true.png" alt="Predicted vs True chart"></p>
<p>The <strong>Residual Histogram</strong> shows the frequency of residual value ranges. Residuals represent variance between predicted and true values that can’t be explained by the model - in other words, errors; so what you should hope to see is that the most frequently occurring residual values are clustered around 0 (in other words, most of the errors are small), with fewer errors at the extreme ends of the scale.</p>
<p><img src="https://docs.microsoft.com/en-us/learn/wwl-data-ai/use-automated-machine-learning/media/residual-histogram.png" alt="Residuals histogram"></p>
<h3 id="deploy-a-model-as-a-service">Deploy a model as a service</h3>
<p>After you’ve used automated machine learning to train some models, you can deploy the best performing model as a service for client applications to use.</p>
<h4 id="deploy-a-predictive-service">Deploy a predictive service</h4>
<p>In Azure Machine Learning, you can deploy a service as an Azure Container Instances (ACI) or to an Azure Kubernetes Service (AKS) cluster. For production scenarios, an AKS deployment is recommended, for which you must create an <em>inference cluster</em> compute target.</p>
<p>You need this information to connect to your deployed service from a client application.</p>
<ul>
<li>The REST endpoint for your service</li>
<li>the Primary Key for your service</li>
</ul>
<h4 id="test-the-deployed-service">Test the deployed service</h4>
<p>E.g. use a Notebook running a Python Script in Azure Machine Learning Studio to consume the created endpoint.</p>
<h3 id="create-a-regression-model-with-azure-machine-learning-designer">Create a Regression Model with Azure Machine Learning designer</h3>
<p><em>Regression</em> is a form of machine learning that is used to predict a numeric <em>label</em> based on an item’s <em>features</em>. For example, an automobile sales company might use the characteristics of a car (such as engine size, number of seats, mileage, and so on) to predict its likely selling price.</p>
<p>Regression is an example of a <em>supervised</em> machine learning technique in which you train a model using data that includes both the features and known values for the label, so that the model learns to <em>fit</em> the feature combinations to the label. Then, after training has been completed, you can use the trained model to predict labels for new items for which the label is unknown.</p>
<p>You can use Microsoft Azure Machine Learning designer to create regression models by using a drag and drop visual interface, without needing to write any code.</p>
<h3 id="create-an-azure-machine-learning-workspace-2">Create an Azure Machine Learning workspace</h3>
<p>see above</p>
<h3 id="create-compute-resources-1">Create compute resources</h3>
<p>see above</p>
<h3 id="explore-data-1">Explore data</h3>
<p>To train a regression model, you need a dataset that includes historical <em>features</em> (characteristics of the entity for which you want to make a prediction) and known <em>label</em> values (the numeric value that you want to train a model to predict).</p>
<h4 id="create-a-pipeline">Create a pipeline</h4>
<p>To use the Azure Machine Learning designer, you create a <em>pipeline</em> that you will use to train a machine learning model. This pipeline starts with the dataset from which you want to train the model. You need to specify a compute target on which to run the pipeline. Remove columns with too many missing values and remaining rows with missing values from the dataset. Mitigate possible bias by <em>normalizing</em> the numeric columns so they’re on the similar scales.</p>
<h4 id="add-data-transformations">Add data transformations</h4>
<p>You typically apply data transformations to prepare the data for modeling. In the case of the automobile price data, you’ll add transformations to address the issues you identified when exploring the data: Select Columns in Dataset, Clean Missing Data, Normalize Data.</p>
<p><img src="https://docs.microsoft.com/en-us/learn/wwl-data-ai/create-regression-model-azure-machine-learning-designer/media/data-transforms.png" alt="Automobile price data (Raw) dataset with Select Columns in Dataset, Clean Missing Data, and Normalize Data modules"></p>
<h4 id="run-the-pipeline">Run the pipeline</h4>
<p>To apply your data transformations, you need to run the pipeline as an experiment.</p>
<h4 id="view-the-transformed-data">View the transformed data</h4>
<p>The dataset is now prepared for model training.</p>
<h3 id="create-and-run-a-training-pipeline">Create and run a training pipeline</h3>
<p>It’s common practice to train the model using a subset of the data, while holding back some data with which to test the trained model. This enables you to compare the labels that the model predicts with the actual known labels in the original dataset.</p>
<p><img src="https://docs.microsoft.com/en-us/learn/wwl-data-ai/create-regression-model-azure-machine-learning-designer/media/train-score.png" alt="split data, then train with linear regression and score"><br>
<img src="https://docs.microsoft.com/en-us/azure/machine-learning/media/algorithm-cheat-sheet/machine-learning-algorithm-cheat-sheet.png" alt="Machine Learning Algorithm Cheat Sheet: Learn how to choose a Machine Learning algorithm."></p>
<h4 id="run-the-training-pipeline">Run the training pipeline</h4>
<p>Now you’re ready to run the training pipeline and train the model.</p>
<p>When the experiment run has completed, select the <strong>Score Model</strong> module and in the settings pane, on the <strong>Outputs + logs</strong> tab, under <strong>Data outputs</strong> in the <strong>Scored dataset</strong> section, use the <strong>Preview Data</strong> icon to view the results.</p>
<h3 id="evaluate-a-regression-model">Evaluate a regression model</h3>
<h4 id="add-an-evaluate-model-module">Add an Evaluate Model module</h4>
<p><img src="https://docs.microsoft.com/en-us/learn/wwl-data-ai/create-regression-model-azure-machine-learning-designer/media/evaluate.png" alt="Evaluate Model module added to Score Model module">When the experiment run has completed, select the <strong>Evaluate Model</strong> module and in the settings pane, on the <strong>Outputs + logs</strong> tab, under <strong>Data outputs</strong> in the <strong>Evaluation results</strong> section, use the <strong>Preview Data</strong> icon to view the results. These include the following regression performance metrics:</p>
<ul>
<li><strong>Mean Absolute Error (MAE)</strong>: The average difference between predicted values and true values. This value is based on the same units as the label, in this case dollars. The lower this value is, the better the model is predicting.</li>
<li><strong>Root Mean Squared Error (RMSE)</strong>: The square root of the mean squared difference between predicted and true values. The result is a metric based on the same unit as the label (dollars). When compared to the MAE (above), a larger difference indicates greater variance in the individual errors (for example, with some errors being very small, while others are large).</li>
<li><strong>Relative Squared Error (RSE)</strong>: A relative metric between 0 and 1 based on the square of the differences between predicted and true values. The closer to 0 this metric is, the better the model is performing. Because this metric is relative, it can be used to compare models where the labels are in different units.</li>
<li><strong>Relative Absolute Error (RAE)</strong>: A relative metric between 0 and 1 based on the absolute differences between predicted and true values. The closer to 0 this metric is, the better the model is performing. Like RSE, this metric can be used to compare models where the labels are in different units.</li>
<li><strong>Coefficient of Determination (R2)</strong>: This metric is more commonly referred to as <em>R-Squared</em>, and summarizes how much of the variance between predicted and true values is explained by the model. The closer to 1 this value is, the better the model is performing.</li>
</ul>
<h3 id="create-an-inference-pipeline">Create an inference pipeline</h3>
<p>After creating and running a pipeline to train the model, you need a second pipeline that performs the same data transformations for new data, and then uses the trained model to <em>infer</em> (in other words, predict) label values based on its features.</p>
<h4 id="create-and-run-an-inference-pipeline">Create and run an inference pipeline</h4>
<p><img src="https://docs.microsoft.com/en-us/learn/wwl-data-ai/create-regression-model-azure-machine-learning-designer/media/inference-changes.png" alt="An inference pipeline with changes indicated"><br>
The inference pipeline assumes that new data will match the schema of the original training data, so the <strong>Automobile price data (Raw)</strong> dataset from the training pipeline is included. However, this input data includes the <strong>price</strong> label that the model predicts, which is unintuitive to include in new car data for which a price prediction has not yet been made. Delete this module and replace it with an <strong>Enter Data Manually</strong> module from the <strong>Data Input and Output</strong> section, containing the following CSV data, which includes feature values without labels for three cars.</p>
<p>Now that you’ve changed the schema of the incoming data to exclude the <strong>price</strong> field, you need to remove any explicit uses of this field in the remaining modules. Select the <strong>Select Columns in Dataset</strong> module and then in the settings pane, edit the columns to remove the <strong>price</strong> field.</p>
<p>The inference pipeline includes the <strong>Evaluate Model</strong> module, which is not useful when predicting from new data, so delete this module.</p>
<h3 id="deploy-a-predictive-service-1">Deploy a predictive service</h3>
<p>After you’ve created and tested an inference pipeline for real-time inferencing, you can publish it as a service for client applications to use.</p>
<h4 id="deploy-a-service">Deploy a service</h4>
<p>deploy a new real-time endpoint</p>
<h4 id="test-the-service">Test the service</h4>
<p>Now you can test your deployed service from a client application</p>
<h2 id="create-a-classification-model-with-azure-machine-learning-designer">Create a classification model with Azure Machine Learning designer</h2>
<p><em>Classification</em> is a form of machine learning that is used to predict which category, or <em>class</em>, an item belongs to.</p>
<p>Classification is an example of a <em>supervised</em> machine learning technique in which you train a model using data that includes both the features and known values for the label, so that the model learns to <em>fit</em> the feature combinations to the label. Then, after training has been completed, you can use the trained model to predict labels for new items for which the label is unknown.</p>
<h3 id="create-an-azure-machine-learning-workspace-3">Create an Azure Machine Learning workspace</h3>
<h3 id="create-compute-resources-2">Create compute resources</h3>
<h3 id="explore-data-2">Explore data</h3>
<h3 id="create-and-run-a-training-pipeline-1">Create and run a training pipeline</h3>
<p><img src="https://docs.microsoft.com/en-us/learn/wwl-data-ai/create-classification-model-azure-machine-learning-designer/media/train-score-pipeline.png" alt="split data, then train with logistic regression and score"></p>
<h3 id="evaluate-a-classification-model">Evaluate a classification model</h3>
<p><img src="https://docs.microsoft.com/en-us/learn/wwl-data-ai/create-classification-model-azure-machine-learning-designer/media/evaluate-pipeline.png" alt="Evaluate Model module added to Score Model module"></p>
<p>View the <em>confusion matrix</em> for the model, which is a tabulation of the predicted and actual value counts for each possible class. For a binary classification model like this one, where you’re predicting one of two possible values, the confusion matrix is a 2x2 grid showing the predicted and actual value counts for classes <strong>0</strong> and <strong>1</strong>, similar to this:</p>
<p><img src="https://docs.microsoft.com/en-us/learn/wwl-data-ai/create-classification-model-azure-machine-learning-designer/media/confusion-matrix.png" alt="A confusion matrix showing actual and predicted value counts for each class">The confusion matrix shows cases where both the predicted and actual values were 1 (known as <em>true positives</em>) at the top left, and cases where both the predicted and the actual values were 0 (<em>true negatives</em>) at the bottom right. The other cells show cases where the predicted and actual values differ (<em>false positives</em> and <em>false negatives</em>). The cells in the matrix are colored so that the more cases represented in the cell, the more intense the color - with the result that you can identify a model that predicts accurately for all classes by looking for a diagonal line of intensely colored cells from the top left to the bottom right (in other words, the cells where the predicted values match the actual values). For a multi-class classification model (where there are more than two possible classes), the same approach is used to tabulate each possible combination of actual and predicted value counts - so a model with three possible classes would result in a 3x3 matrix with a diagonal line of cells where the predicted and actual labels match.</p>
<p>Metrics of the confusion matrix:</p>
<ul>
<li><strong>Accuracy</strong>: The ratio of correct predictions (true positives + true negatives) to the total number of predictions. In other words, what proportion of diabetes predictions did the model get right?</li>
<li><strong>Precision</strong>: The fraction of positive cases correctly identified (the number of true positives divided by the number of true positives plus false positives). In other words, out of all the patients that the model predicted as having diabetes, how many are actually diabetic?</li>
<li><strong>Recall</strong>: The fraction of the cases classified as positive that are actually positive (the number of true positives divided by the number of true positives plus false negatives). In other words, out of all the patients who actually have diabetes, how many did the model identify?</li>
<li><strong>F1 Score</strong>: An overall metric that essentially combines precision and recall.</li>
</ul>
<p>Of these metric, <em>accuracy</em> is the most intuitive. However, you need to be careful about using simple accuracy as a measurement of how well a model works. Suppose that only 3% of the population is diabetic. You could create a model that always predicts <strong>0</strong> and it would be 97% accurate - just not very useful! For this reason, most data scientists use other metrics like precision and recall to assess classification model performance.</p>
<p>Above the list of metrics, note that there’s a <strong>Threshold</strong> slider. Remember that what a classification model predicts is the probability for each possible class. In the case of this binary classification model, the predicted probability for a <em>positive</em> (that is, diabetic) prediction is a value between 0 and 1. By default, a predicted probability for diabetes <em>including or above</em> 0.5 results in a class prediction of 1, while a prediction <em>below</em> this threshold means that there’s a greater probability of the patient <strong>not</strong> having diabetes (remember that the probabilities for all classes add up to 1), so the predicted class would be 0.</p>
<p><strong>ROC curve</strong><br>
ROC stands for <em>received operator characteristic</em>, but most data scientists just call it a ROC curve. Another term for <em>recall</em> is <strong>True positive rate</strong>, and it has a corresponding metric named <strong>False positive rate</strong>, which measures the number of negative cases incorrectly identified as positive compared the number of actual negative cases.</p>
<p><strong>AUC</strong><br>
Plotting these metrics against each other for every possible threshold value between 0 and 1 results in a curve. In an ideal model, the curve would go all the way up the left side and across the top, so that it covers the full area of the chart. The larger the <em>area under the curve</em> (which can be any value from 0 to 1), the better the model is performing. To get an idea of how this area represents the performance of the model, imagine a straight diagonal line from the bottom left to the top right of the ROC chart. This represents the expected performance if you just guessed or flipped a coin for each patient - you could expect to get around half of them right, and half of them wrong, so the area under the diagonal line represents an AUC of 0.5. If the AUC for your model is higher than this for a binary classification model, then the model performs better than a random guess.</p>
<h3 id="create-an-inference-pipeline-1">Create an inference pipeline</h3>
<p><img src="https://docs.microsoft.com/en-us/learn/wwl-data-ai/create-classification-model-azure-machine-learning-designer/media/inference-changes.png" alt="An inference pipeline with changes indicated"></p>
<h3 id="deploy-a-predictive-service-2">Deploy a predictive service</h3>
<h2 id="create-a-clustering-model-with-azure-machine-learning-designer">Create a Clustering Model with Azure Machine Learning designer</h2>
<p><em>Clustering</em> is a form of machine learning that is used to group similar items into clusters based on their features. For example, a researcher might take measurements of penguins, and group them based on similarities in their proportions.</p>
<p>Clustering is an example of <em>unsupervised</em> machine learning, in which you train a model to separate items into clusters based purely on their characteristics, or <em>features</em>. There is no previously known cluster value (or <em>label</em>) from which to train the model.</p>
<h3 id="explore-data-3">Explore data</h3>
<h3 id="create-and-run-a-training-pipeline-2">Create and run a training pipeline</h3>
<p><img src="https://docs.microsoft.com/en-us/learn/wwl-data-ai/create-clustering-model-azure-machine-learning-designer/media/k-means.png" alt="split data, then use the K-Means Clustering algorithm to train a model and the Assign Data to Modules module to test it"><br>
The <em>K-Means</em> algorithm groups items into the number of clusters you specify - a value referred to as <em><strong>K</strong></em>.</p>
<p>You can think of data observations, like the penguin measurements, as being multidimensional vectors. The K-Means algorithm works by:</p>
<ol>
<li>initializing <em>K</em> coordinates as randomly selected points called <em>centroids</em> in <em>n</em>-dimensional space (where <em>n</em> is the number of dimensions in the feature vectors).</li>
<li>Plotting the feature vectors as points in the same space, and assigning each point to its closest centroid.</li>
<li>Moving the centroids to the middle of the points allocated to it (based on the <em>mean</em> distance).</li>
<li>Reassigning the points to their closest centroid after the move.</li>
<li>Repeating steps 3 and 4 until the cluster allocations stabilize or the specified number of iterations has completed.</li>
</ol>
<h3 id="evaluate-a-clustering-model">Evaluate a clustering model</h3>
<p>Evaluating a clustering model is made difficult by the fact that there are no previously known <em>true</em> values for the cluster assignments. A successful clustering model is one that achieves a good level of separation between the items in each cluster, so we need metrics to help us measure that separation.</p>
<p><img src="https://docs.microsoft.com/en-us/learn/wwl-data-ai/create-clustering-model-azure-machine-learning-designer/media/evaluate-cluster.png" alt="Evaluate Model module added to Assign Data to Clusters module"><br>
The metrics in each row are:</p>
<ul>
<li><strong>Average Distance to Other Center</strong>: This indicates how close, on average, each point in the cluster is to the centroids of all other clusters.</li>
<li><strong>Average Distance to Cluster Center</strong>: This indicates how close, on average, each point in the cluster is to the centroid of the cluster.</li>
<li><strong>Number of Points</strong>: The number of points assigned to the cluster.</li>
<li><strong>Maximal Distance to Cluster Center</strong>: The maximum of the distances between each point and the centroid of that point’s cluster. If this number is high, the cluster may be widely dispersed. This statistic in combination with the <strong>Average Distance to Cluster Center</strong> helps you determine the cluster’s <em>spread</em>.</li>
</ul>
<h3 id="create-an-inference-pipeline-2">Create an inference pipeline</h3>
<p><img src="https://docs.microsoft.com/en-us/learn/wwl-data-ai/create-clustering-model-azure-machine-learning-designer/media/inference-changes.png" alt="Replace penguin-data dataset with Enter Data Manually module. Remove Select Columns in Dataset and Evaluate Model modules"></p>
<h3 id="deploy-a-predictive-service-3">Deploy a predictive service</h3>
<p>Use the <strong>predict-penguin-clusters</strong> service you created to predict a cluster assignment.</p>
<h1 id="explore-computer-vision-in-microsoft-azure">Explore computer vision in Microsoft Azure</h1>
<h2 id="analyze-images-with-the-computer-vision-service">Analyze images with the Computer Vision service</h2>
<p><em>Computer Vision</em> is a branch of artificial intelligence (AI) that explores the development of AI systems that can “see” the world, either in real-time through a camera or by analyzing images and video. This is made possible by the fact that digital images are essentially just arrays of numeric pixel values, and we can use those pixel values as <em>features</em> to train machine learning models that can classify images, detect discrete objects in an image, and even generate text-based summaries of photographs.</p>
<p>Some potential uses for computer vision include:</p>
<ul>
<li>
<p><strong>Content Organization</strong>: Identify people or objects in photos and organize them based on that identification. Photo recognition applications like this are commonly used in photo storage and social media applications.</p>
</li>
<li>
<p><strong>Text Extraction</strong>: Analyze images and PDF documents that contain text and extract the text into a structured format.</p>
</li>
<li>
<p><strong>Spatial Analysis</strong>: Identify people or objects, such as cars, in a space and map their movement within that space.</p>
</li>
</ul>
<h3 id="get-started-with-image-analysis-on-azure">Get started with image analysis on Azure</h3>
<h4 id="azure-resources-for-computer-vision">Azure resources for Computer Vision</h4>
<p>In Microsoft Azure, the <strong>Computer Vision</strong> cognitive service uses pre-trained models to analyze images, enabling software developers to easily build applications that can:</p>
<ul>
<li>Interpret an image and suggest an appropriate caption.</li>
<li>Suggest relevant <em>tags</em> that could be used to index an image.</li>
<li>Categorize an image.</li>
<li>Identify objects in an image.</li>
<li>Detect faces and people in an image.</li>
<li>Recognize celebrities and landmarks in an image.</li>
<li>Read text in an image.</li>
</ul>
<p>To use the Computer Vision service, you need to create a resource for it in your Azure subscription. You can use either of the following resource types:</p>
<ul>
<li><strong>Computer Vision</strong>: A specific resource for the Computer Vision service. Use this resource type if you don’t intend to use any other cognitive services, or if you want to track utilization and costs for your Computer Vision resource separately.</li>
<li><strong>Cognitive Services</strong>: A general cognitive services resource that includes Computer Vision along with many other cognitive services; such as Text Analytics, Translator Text, and others. Use this resource type if you plan to use multiple cognitive services and want to simplify administration and development.</li>
<li></li>
</ul>
<p>Whichever type of resource you choose to create, it will provide two pieces of information that you will need to use it:</p>
<ul>
<li>A <strong>key</strong> that is used to authenticate client applications.</li>
<li>An <strong>endpoint</strong> that provides the HTTP address at which your resource can be accessed.</li>
</ul>
<h4 id="analyzing-images-with-the-computer-vision-service">Analyzing images with the Computer Vision service</h4>
<h5 id="describing-an-image">Describing an image</h5>
<p>Computer Vision has the ability to analyze an image, evaluate the objects that are detected, and generate a human-readable phrase or sentence that can describe what was detected in the image. Depending on the image contents, the service may return multiple results, or phrases. Each returned phrase will have an associated confidence score, indicating how confident the algorithm is in the supplied description. The highest confidence phrases will be listed first.</p>
<h5 id="tagging-visual-features">Tagging visual features</h5>
<p>The image descriptions generated by Computer Vision are based on a set of thousands of recognizable objects, which can be used to suggest <em>tags</em> for the image. These tags can be associated with the image as metadata that summarizes attributes of the image; and can be particularly useful if you want to index an image along with a set of key terms that might be used to search for images with specific attributes or contents.</p>
<h5 id="detecting-objects">Detecting objects</h5>
<p>The object detection capability is similar to tagging, in that the service can identify common objects; but rather than tagging, or providing tags for the recognized objects only, this service can also return what is known as bounding box coordinates. Not only will you get the type of object, but you will also receive a set of coordinates that indicate the top, left, width, and height of the object detected, which you can use to identify the location of the object in the image.</p>
<h5 id="detecting-brands">Detecting brands</h5>
<p>This feature provides the ability to identify commercial brands. The service has an existing database of thousands of globally recognized logos from commercial brands of products. If a known brand is detected, the service returns a response that contains the brand name, a confidence score (from 0 to 1 indicating how positive the identification is), and a bounding box (coordinates) for where in the image the detected brand was found.</p>
<h5 id="detecting-faces">Detecting faces</h5>
<p>The Computer Vision service can detect and analyze human faces in an image, including the ability to determine age and a bounding box rectangle for the location of the face(s). The facial analysis capabilities of the Computer Vision service are a subset of those provided by the dedicated <a href="https://docs.microsoft.com/en-us/azure/cognitive-services/face/">Face Service</a>. If you need basic face detection and analysis, combined with general image analysis capabilities, you can use the Computer Vision service; but for more comprehensive facial analysis and facial recognition functionality, use the Face service.</p>
<h5 id="categorizing-an-image">Categorizing an image</h5>
<p>Computer Vision can categorize images based on their contents. The service uses a parent/child hierarchy with a “current” limited set of categories. When analyzing an image, detected objects are compared to the existing categories to determine the best way to provide the categorization.</p>
<h5 id="detecting-domain-specific-content">Detecting domain-specific content</h5>
<p>When categorizing an image, the Computer Vision service supports two specialized domain models:</p>
<ul>
<li><strong>Celebrities</strong> - The service includes a model that has been trained to identify thousands of well-known celebrities from the worlds of sports, entertainment, and business.</li>
<li><strong>Landmarks</strong> - The service can identify famous landmarks, such as the Taj Mahal and the Statue of Liberty.</li>
</ul>
<h5 id="optical-character-recognition">Optical character recognition</h5>
<p>The Computer Vision service can use optical character recognition (OCR) capabilities to detect printed and handwritten text in images.</p>
<h5 id="additional-capabilities">Additional capabilities</h5>
<ul>
<li>Detect image types - for example, identifying clip art images or line drawings.</li>
<li>Detect image color schemes - specifically, identifying the dominant foreground, background, and overall colors in an image.</li>
<li>Generate thumbnails - creating small versions of images.</li>
<li>Moderate content - detecting images that contain adult content or depict violent, gory scenes.</li>
</ul>
<h2 id="classify-images-with-the-custom-vision-service">Classify images with the Custom Vision service</h2>
<p>Image classification is a common workload in artificial intelligence (AI) applications. It harnesses the predictive power of machine learning to enable AI systems to identify real-world items based on images.</p>
<p>Some potential uses for image classification include:</p>
<ul>
<li><strong>Product identification</strong>: performing visual searches for specific products in online searches or even, in-store using a mobile device.</li>
<li><strong>Disaster investigation</strong>: identifying key infrastructure for major disaster preparation efforts. For example, identifying bridges and roads in aerial images can help disaster relief teams plan ahead in regions that are not well mapped.</li>
<li><strong>Medical diagnosis</strong>: evaluating images from X-ray or MRI devices could quickly classify specific issues found as cancerous tumors, or many other medical conditions related to medical imaging diagnosis.</li>
<li><strong>Anomaly detection</strong>: performing visual detection for defects or anomalies in products during manufacturing processes.</li>
</ul>
<p>Training a machine learning model to classify images from scratch requires considerable time, deep learning expertise, and data. Microsoft’s Custom Vision service enables you to build image classification models that can be deployed as AI solutions. Using the Custom Vision service requires less time and machine learning expertise than building from scratch.</p>
<p>The <em>Computer Vision</em> cognitive service provides useful pre-built models for working with images, but you’ll often need to train your own model for computer vision.</p>
<p>In Azure, you can use the <strong><em>Custom Vision</em></strong> cognitive service to train an image classification model based on existing images. There are two elements to creating an image classification solution. First, you must train a model to recognize different classes using existing images. Then, when the model is trained you must publish it as a service that can be consumed by applications.</p>
<h3 id="understand-classification">Understand classification</h3>
<p>You can use a machine learning <em>classification</em> technique to predict which category, or <em>class</em>, something belongs to. Classification machine learning models use a set of inputs, which we call <em>features</em>, to calculate a probability score for each possible class and predict a <em>label</em> that indicates the most likely class that an object belongs to.</p>
<h4 id="understand-image-classification">Understand image classification</h4>
<p><em>Image classification</em> is a machine learning technique in which the object being classified is an image, such as a photograph.</p>
<p>To create an image classification model, you need data that consists of features and their labels. The existing data is a set of categorized images. Digital images are made up of an array of pixel values, and these are used as features to train the model based on the known image classes.</p>
<h4 id="azures-custom-vision-service">Azure’s Custom Vision service</h4>
<p>Most modern image classification solutions are based on <em>deep learning</em> techniques that make use of <em>convolutional neural networks</em> (CNNs) to uncover patterns in the pixels that correspond to particular classes.<br>
Common techniques used to train image classification models have been encapsulated into the <strong>Custom Vision</strong> cognitive service in Microsoft Azure; making it easy to train a model and publish it as a software service with minimal knowledge of deep learning techniques.</p>
<h3 id="get-started-with-image-classification-on-azure">Get started with image classification on Azure</h3>
<h4 id="azure-resources-for-custom-vision">Azure resources for Custom Vision</h4>
<p>Creating an image classification solution with Custom Vision consists of two main tasks. First you must use existing images to train the model, and then you must publish the model so that client applications can use it to generate predictions.</p>
<h4 id="model-training">Model training</h4>
<p>To train a classification model, you must upload images to your training resource and label them with the appropriate class labels. Then, you must train the model and evaluate the training results.</p>
<p>You can perform these tasks in the <a href="https://www.customvision.ai"><em>Custom Vision portal</em></a>, or if you have the necessary coding experience you can use one of the Custom Vision service programming language-specific <a href="https://docs.microsoft.com/en-us/azure/cognitive-services/Custom-Vision-Service/quickstarts/image-classification">software development kits (SDKs)</a>.</p>
<h4 id="model-evaluation">Model evaluation</h4>
<p>Model training process is an iterative process in which the Custom Vision service repeatedly trains the model using some of the data, but holds some back to evaluate the model. At the end of the training process, the performance for the trained model is indicated by the following evaluation metrics:</p>
<ul>
<li><strong>Precision</strong>: What percentage of the class predictions made by the model were correct? For example, if the model predicted that 10 images are oranges, of which eight were actually oranges, then the precision is 0.8 (80%).</li>
<li><strong>Recall</strong>: What percentage of class predictions did the model correctly identify? For example, if there are 10 images of apples, and the model found 7 of them, then the recall is 0.7 (70%).</li>
<li><strong>Average Precision (AP)</strong>: An overall metric that takes into account both precision and recall).</li>
</ul>
<p>One way to improve the performance of your model is to add more images to the training set.</p>
<h4 id="using-the-model-for-prediction">Using the model for prediction</h4>
<p>After you’ve trained the model, and you’re satisfied with its evaluated performance, you can publish the model to your prediction resource.</p>
<p>To use your model, client application developers need the following information:</p>
<ul>
<li><strong>Project ID</strong>: The unique ID of the Custom Vision project you created to train the model.</li>
<li><strong>Model name</strong>: The name you assigned to the model during publishing.</li>
<li><strong>Prediction endpoint</strong>: The HTTP address of the endpoints for the <em>prediction</em> resource to which you published the model (<em><strong>not</strong></em> the training resource).</li>
<li><strong>Prediction key</strong>: The authentication key for the <em>prediction</em> resource to which you published the model (<em><strong>not</strong></em> the training resource).</li>
</ul>
<h2 id="detect-objects-in-images-with-the-custom-vision-service">Detect objects in images with the Custom Vision service</h2>
<p><em>Object detection</em> is a form of machine learning based computer vision in which a model is trained to recognize individual types of object in an image, and to identify their location in the image.</p>
<p>Notice that an object detection model returns the following information:</p>
<ul>
<li>The <em>class</em> of each object identified in the image.</li>
<li>The probability score of the object classification (which you can interpret as the <em>confidence</em> of the predicted class being correct)</li>
<li>The coordinates of a <em>bounding box</em> for each object.</li>
</ul>
<h3 id="get-started-with-object-detection-on-azure">Get started with object detection on Azure</h3>
<p>The <strong>Custom Vision</strong> cognitive service in Azure enables you to create object detection models that meet the needs of many computer vision scenarios with minimal deep learning expertise and fewer training images.</p>
<h4 id="azure-resources-for-custom-vision-1">Azure resources for Custom Vision</h4>
<p>If you choose to create a Custom Vision resource, you will be prompted to choose <em>training</em>, <em>prediction</em>, or <em>both</em> - and it’s important to note that if you choose “both”, then <em><strong>two</strong></em> resources are created - one for training and one for prediction.</p>
<p>It’s also possible to take a mix-and-match approach in which you use a dedicated Custom Vision resource for training, but deploy your model to a Cognitive Services resource for prediction. For this to work, the training and prediction resources must be created in the same region.</p>
<h4 id="image-tagging">Image tagging</h4>
<p>Before you can train an object detection model, you must tag the classes and bounding box coordinates in a set of training images. This process can be time-consuming, but the <em>Custom Vision portal</em> provides a graphical interface that makes it straightforward.</p>
<h4 id="model-training-and-evaluation">Model training and evaluation</h4>
<p>At the end of the training process, the performance for the trained model is indicated by the following evaluation metrics:</p>
<ul>
<li><strong>Precision</strong>: What percentage of class predictions did the model correctly identify? For example, if the model predicted that 10 images are oranges, of which eight were actually oranges, then the precision is 0.8 (80%).</li>
<li><strong>Recall</strong>: What percentage of the class predictions made by the model were correct? For example, if there are 10 images of apples, and the model found 7 of them, then the recall is 0.7 (70%).</li>
<li><strong>Mean Average Precision (mAP)</strong>: An overall metric that takes into account both precision and recall across all classes).</li>
</ul>
<h2 id="detect-and-analyze-faces-with-the-face-service">Detect and analyze faces with the Face service</h2>
<p>Face detection and analysis is an area of artificial intelligence (AI) in which we use algorithms to locate and analyze human faces in images or video content.</p>
<h3 id="introduction">Introduction</h3>
<p>Face detection and analysis is an area of artificial intelligence (AI) in which we use algorithms to locate and analyze human faces in images or video content.</p>
<h4 id="face-detection">Face detection</h4>
<p>Face detection involves identifying regions of an image that contain a human face, typically by returning <em>bounding box</em> coordinates that form a rectangle around the face, like this:</p>
<p><img src="https://docs.microsoft.com/en-us/learn/wwl-data-ai/detect-analyze-faces/media/face-detection.png" alt="An image with two faces highlighted in rectangles"></p>
<h4 id="facial-analysis">Facial analysis</h4>
<p>Moving beyond simple face detection, some algorithms can also return other information, such as facial landmarks (nose, eyes, eyebrows, lips, and others).</p>
<p><img src="https://docs.microsoft.com/en-us/learn/wwl-data-ai/detect-analyze-faces/media/landmarks-1.png" alt="facial landmarks image showing data around face characteristics"></p>
<p>These facial landmarks can be used as features with which to train a machine learning model from which you can infer information about a person, such as their perceived age or perceived emotional state, like this:</p>
<p><img src="https://docs.microsoft.com/en-us/learn/wwl-data-ai/detect-analyze-faces/media/face-attributes.png" alt="A happy 25-year old"></p>
<h4 id="facial-recognition">Facial recognition</h4>
<p>A further application of facial analysis is to train a machine learning model to identify known individuals from their facial features. This usage is more generally known as <em>facial recognition</em>, and involves using multiple images of each person you want to recognize to train a model so that it can detect those individuals in new images on which it wasn’t trained.</p>
<p><img src="https://docs.microsoft.com/en-us/learn/wwl-data-ai/detect-analyze-faces/media/facial-recognition.png" alt="A person identified as &quot;Wendell&quot;"></p>
<h4 id="uses-of-face-detection-and-analysis">Uses of face detection and analysis</h4>
<p>There are many applications for face detection, analysis, and recognition. For example,</p>
<ul>
<li>Security - facial recognition can be used in building security applications, and increasingly it is used in smart phones operating systems for unlocking devices.</li>
<li>Social media - facial recognition can be used to automatically tag known friends in photographs.</li>
<li>Intelligent monitoring - for example, an automobile might include a system that monitors the driver’s face to determine if the driver is looking at the road, looking at a mobile device, or shows signs of tiredness.</li>
<li>Advertising - analyzing faces in an image can help direct advertisements to an appropriate demographic audience.</li>
<li>Missing persons - using public cameras systems, facial recognition can be used to identify if a missing person is in the image frame.</li>
<li>Identity validation - useful at ports of entry kiosks where a person holds a special entry permit.</li>
</ul>
<h3 id="get-started-with-face-analysis-on-azure">Get started with Face analysis on Azure</h3>
<p>Microsoft Azure provides multiple cognitive services that you can use to detect and analyze faces, including:</p>
<ul>
<li><strong>Computer Vision</strong>, which offers face detection and some basic face analysis, such as determining age.</li>
<li><strong>Video Indexer</strong>, which you can use to detect and identify faces in a video.</li>
<li><strong>Face</strong>, which offers pre-built algorithms that can detect, recognize, and analyze faces.</li>
</ul>
<h2 id="face">Face</h2>
<p>Face currently supports the following functionality:</p>
<ul>
<li>Face Detection</li>
<li>Face Verification</li>
<li>Find Similar Faces</li>
<li>Group faces based on similarities</li>
<li>Identify people</li>
</ul>
<p>Face can return the rectangle coordinates for any human faces that are found in an image, as well as a series of attributes related to those faces such as:</p>
<ul>
<li><strong>Age</strong></li>
<li><strong>Blur</strong></li>
<li><strong>Emotion</strong></li>
<li><strong>Exposure</strong></li>
<li><strong>Facial hair</strong></li>
<li><strong>Glasses</strong></li>
<li><strong>Hair</strong></li>
<li><strong>Head pose</strong></li>
<li><strong>Makeup</strong></li>
<li><strong>Noise</strong></li>
<li><strong>Occlusion</strong></li>
<li><strong>Smile</strong></li>
</ul>
<h4 id="tips-for-more-accurate-results">Tips for more accurate results</h4>
<p>There are some considerations that can help improve the accuracy of the detection in the images:</p>
<ul>
<li>image format - supported images are JPEG, PNG, GIF, and BMP</li>
<li>file size - 6 MB or smaller</li>
<li>face size range - from 36 x 36 up to 4096 x 4096. Smaller or larger faces will not be detected</li>
<li>other issues - face detection can be impaired by extreme face angles, occlusion (objects blocking the face such as sunglasses or a hand). Best results are obtained when the faces are full-frontal or as near as possible to full-frontal.</li>
</ul>
<h3 id="read-text-with-the-computer-vision-service">Read text with the Computer Vision service</h3>
<p>The ability for computer systems to process written or printed text is an area of artificial intelligence (AI) where <em>computer vision</em> intersects with <em>natural language processing</em>. You need computer vision capabilities to “read” the text, and then you need natural language processing capabilities to make sense of it.</p>
<p>The basic foundation of processing printed text is <em>optical character recognition</em> (OCR), in which a model can be trained to recognize individual shapes as letters, numerals, punctuation, or other elements of text. It’s now even possible to build models that can detect printed or handwritten text in an image and read it line-by-line or even word-by-word.</p>
<p>At the other end of the scale, there is <em>machine reading comprehension</em> (MRC), in which an AI system not only reads the text characters, but can use a semantic model to interpret what the text is about.</p>
<p>The ability to recognize printed and handwritten text in images, is beneficial in many scenarios such as:</p>
<ul>
<li>note taking</li>
<li>digitizing forms, such as medical records or historical documents</li>
<li>scanning printed or handwritten checks for bank deposits</li>
</ul>
<h3 id="get-started-with-ocr-on-azure">Get started with OCR on Azure</h3>
<h4 id="use-the-computer-vision-service-to-read-text">Use the Computer Vision service to read text</h4>
<p>The Computer Vision service provides two application programming interfaces (APIs) that you can use to read text in images: the <strong>OCR</strong> API and the <strong>Read</strong> API.</p>
<h3 id="the-ocr-api">The OCR API</h3>
<p>The OCR API is designed for quick extraction of small amounts of text in images. It operates synchronously to provide immediate results, and can recognize text in numerous languages.</p>
<p>When you use the OCR API to process an image, it returns a hierarchy of information that consists of:</p>
<ul>
<li><strong>Regions</strong> in the image that contain text</li>
<li><strong>Lines</strong> of text in each region</li>
<li><strong>Words</strong> in each line of text</li>
</ul>
<p>For each of these elements, the OCR API also returns <em>bounding box</em> coordinates that define a rectangle to indicate the location in the image where the region, line, or word appears.</p>
<h3 id="the-read-api">The Read API</h3>
<p>The OCR method can have issues with false positives when the image is considered text-dominate. The Read API uses the latest recognition models and is optimized for images that have a significant amount of text or has considerable visual noise.</p>
<p>Because the Read API can work with larger documents, it works asynchronously so as not to block your application while it is reading the content and returning results to your application. This means that to use the Read API, your application must use a three-step process:</p>
<ol>
<li>Submit an image to the API, and retrieve an <em>operation ID</em> in response.</li>
<li>Use the operation ID to check on the status of the image analysis operation, and wait until it has completed.</li>
<li>Retrieve the results of the operation.</li>
</ol>
<p>The results from the Read API are arranged into the following hierarchy:</p>
<ul>
<li><strong>Pages</strong> - One for each page of text, including information about the page size and orientation.</li>
<li><strong>Lines</strong> - The lines of text on a page.</li>
<li><strong>Words</strong> - The words in a line of text.</li>
</ul>
<p>Each line and word includes bounding box coordinates indicating its position on the page.</p>
<h2 id="analyze-receipts-with-the-form-recognizer-service">Analyze receipts with the Form Recognizer service</h2>
<p>A common problem in many organizations is the need to process receipt or invoice data. It’s relatively easy to scan receipts to create digital images or PDF documents, and it’s possible to use optical character recognition (OCR) technologies to extract the text contents from the digitized documents. However, typically someone still needs to review the extracted text to make sense of the information it contains.</p>
<p>Increasingly, organizations with large volumes of receipts and invoices to process are looking for artificial intelligence (AI) solutions that can not only extract the text data from receipts, but also intelligently interpret the information they contain.</p>
<h3 id="get-started-with-receipt-analysis-on-azure">Get started with receipt analysis on Azure</h3>
<p>The <strong>Form Recognizer</strong> in Azure provides intelligent form processing capabilities that you can use to automate the processing of data in documents such as forms, invoices, and receipts. It combines state-of-the-art optical character recognition (OCR) with predictive models that can interpret form data by:</p>
<ul>
<li>Matching field names to values.</li>
<li>Processing tables of data.</li>
<li>Identifying specific types of field, such as dates, telephone numbers, addresses, totals, and others.</li>
</ul>
<p>Form Recognizer supports automated document processing through:</p>
<ul>
<li><strong>A pre-built receipt model</strong> that is provided out-of-the-box, and is trained to recognize and extract data from sales receipts.</li>
<li><strong>Custom models</strong>, which enable you to extract what are known as key/value pairs and table data from forms. Custom models are trained using your own data, which helps to tailor this model to your specific forms.</li>
</ul>
<h4 id="using-the-pre-built-receipt-model">Using the pre-built receipt model</h4>
<p>Currently the pre-built receipt model is designed to recognize common receipts, in English, that are common to the USA. Examples are receipts used at restaurants, retail locations, and gas stations. The model is able to extract key information from the receipt slip:</p>
<ul>
<li>time of transaction</li>
<li>date of transaction</li>
<li>merchant information</li>
<li>taxes paid</li>
<li>receipt totals</li>
<li>other pertinent information that may be present on the receipt</li>
<li>all text on the receipt is recognized and returned as well</li>
</ul>
<p>Use the following guidelines to get the best results when using a custom model.</p>
<ul>
<li>Images must be JPEG, PNG, BMP, PDF, or TIFF formats</li>
<li>File size must be less than 50 MB</li>
<li>Image size between 50 x 50 pixels and 10000 x 10000 pixels</li>
<li>For PDF documents, no larger than 17 inches x 17 inches</li>
</ul>
<h1 id="explore-natural-language-processing">Explore natural language processing</h1>
<h2 id="analyze-text-with-the-text-analytics-service">Analyze text with the Text Analytics service</h2>
<p>Analyzing text is a process where you evaluate different aspects of a document or phrase, in order to gain insights into the content of that text. For the most part, humans are able to read some text and understand the meaning behind it.</p>
<h3 id="text-analytics-techniques">Text Analytics Techniques</h3>
<p>Text analytics is a process where an artificial intelligence (AI) algorithm, running on a computer, evaluates these same attributes in text, to determine specific insights. A person will typically rely on their own experiences and knowledge to achieve the insights. A computer must be provided with similar knowledge to be able to perform the task. There are some commonly used techniques that can be used to build software to analyze text, including:</p>
<ul>
<li>Statistical analysis of terms used in the text.</li>
<li>Extending frequency analysis to multi-term phrases, commonly known as <em>N-grams</em></li>
<li>Applying <em>stemming</em> or <em>lemmatization</em> algorithms to normalize words before counting them</li>
<li>Applying linguistic structure rules to analyze sentences - for example, breaking down sentences into tree-like structures such as a <em>noun phrase</em>, which itself contains <em>nouns</em>, <em>verbs</em>, <em>adjectives</em>, and so on.</li>
<li>Encoding words or terms as numeric features that can be used to train a machine learning model. For example, to classify a text document based on the terms it contains. This technique is often used to perform <em>sentiment analysis</em>, in which a document is classified as positive or negative.</li>
<li>Creating <em>vectorized</em> models that capture semantic relationships between words by assigning them to locations in n-dimensional space.</li>
</ul>
<p>In Microsoft Azure, the <strong>Text Analytics</strong> cognitive service can help simplify application development by using pre-trained models that can:</p>
<ul>
<li>Determine the language of a document or text (for example, French or English).</li>
<li>Perform sentiment analysis on text to determine a positive or negative sentiment.</li>
<li>Extract key phrases from text that might indicate its main talking points.</li>
<li>Identify and categorize entities in the text. Entities can be people, places, organizations, or even everyday items such as dates, times, quantities, and so on.</li>
</ul>
<p>In this module, you’ll explore some of these capabilities and gain an understanding of how you might apply them to applications such as:</p>
<ul>
<li>A social media feed analyzer to detect sentiment around a political campaign or a product in market.</li>
<li>A document search application that extracts key phrases to help summarize the main subject matter of documents in a catalog.</li>
<li>A tool to extract brand information or company names from documents or other text for identification purposes.</li>
</ul>
<h3 id="get-started-with-text-analytics-on-azure">Get started with Text Analytics on Azure</h3>
<p>The Text Analytics service is a part of the Azure Cognitive Services offerings that can perform advanced natural language processing over raw text.</p>
<h4 id="language-detection">Language detection</h4>
<p>Use the language detection capability of the Text Analytics service to identify the language in which text is written. You can submit multiple documents at a time for analysis. For each document submitted to it, the service will detect:</p>
<ul>
<li>The language name (for example “English”).</li>
<li>The ISO 6391 language code (for example, “en”).</li>
<li>A score indicating a level of confidence in the language detection.</li>
</ul>
<p>The language detection service will focus on the <em><strong>predominant</strong></em> language in the text. The service uses an algorithm to determine the predominant language, such as length of phrases or total amount of text for the language compared to other languages in the text.</p>
<h5 id="ambiguous-or-mixed-language-content">Ambiguous or mixed language content</h5>
<p>There may be text that is ambiguous in nature, or that has mixed language content. These situations can present a challenge to the service. An ambiguous content example would be a case where the document contains limited text, or only punctuation. For example, using the service to analyze the text “:-)”, results in a value of <strong>unknown</strong> for the language name and the language identifier, and a score of <strong>NaN</strong> (which is used to indicate <em>not a number</em>).</p>
<h4 id="sentiment-analysis">Sentiment analysis</h4>
<p>The Text Analytics service can evaluate text and return sentiment scores and labels for each sentence. This capability is useful for detecting positive and negative sentiment in social media, customer reviews, discussion forums and more.</p>
<p>Using the pre-built machine learning classification model, the service evaluates the text and returns a sentiment score in the range of 0 to 1, with values closer to 1 being a positive sentiment. Scores that are close to the middle of the range (0.5) are considered neutral or indeterminate.</p>
<h5 id="indeterminate-sentiment">Indeterminate sentiment</h5>
<p>A score of 0.5 might indicate that the sentiment of the text is indeterminate, and could result from text that does not have sufficient context to discern a sentiment or insufficient phrasing. For example, a list of words in a sentence that has no structure, could result in an indeterminate score.</p>
<h4 id="key-phrase-extraction">Key phrase extraction</h4>
<p>Key phrase extraction is the concept of evaluating the text of a document, or documents, and then identifying the main talking points of the document(s).</p>
<h4 id="entity-recognition">Entity recognition</h4>
<p>You can provide the Text Analytics service with unstructured text and it will return a list of <em>entities</em> in the text that it recognizes. An entity is essentially an item of a particular type or a category; and in some cases, subtype. For recognized entities, the service returns a URL for a relevant <em>Wikipedia</em> article.</p>
<h2 id="recognize-and-synthesize-speech">Recognize and synthesize speech</h2>
<p>Increasingly, we expect artificial intelligence (AI) solutions to accept vocal commands and provide spoken responses.</p>
<p>To enable this kind of interaction, the AI system must support two capabilities:</p>
<ul>
<li><strong>Speech recognition</strong> - the ability to detect and interpret spoken input.</li>
<li><strong>Speech synthesis</strong> - the ability to generate spoken output.</li>
</ul>
<h4 id="speech-recognition">Speech recognition</h4>
<p>Speech recognition is concerned with taking the spoken word and converting it into data that can be processed - often by transcribing it into a text representation. The spoken words can be in the form of a recorded voice in an audio file, or live audio from a microphone. Speech patterns are analyzed in the audio to determine recognizable patterns that are mapped to words. To accomplish this feat, the software typically uses multiple types of model, including:</p>
<ul>
<li>An <em>acoustic</em> model that converts the audio signal into phonemes (representations of specific sounds).</li>
<li>A <em>language</em> model that maps phonemes to words, usually using a statistical algorithm that predicts the most probable sequence of words based on the phonemes.</li>
</ul>
<p>The recognized words are typically converted to text, which you can use for various purposes, such as.</p>
<ul>
<li>Providing closed captions for recorded or live videos</li>
<li>Creating a transcript of a phone call or meeting</li>
<li>Automated note dictation</li>
<li>Determining intended user input for further processing</li>
</ul>
<h4 id="speech-synthesis">Speech synthesis</h4>
<p>Speech synthesis is in many respects the reverse of speech recognition. It is concerned with vocalizing data, usually by converting text to speech. A speech synthesis solution typically requires the following information:</p>
<ul>
<li>The text to be spoken.</li>
<li>The voice to be used to vocalize the speech.</li>
</ul>
<p>To synthesize speech, the system typically <em>tokenizes</em> the text to break it down into individual words, and assigns phonetic sounds to each word. It then breaks the phonetic transcription into <em>prosodic</em> units (such as phrases, clauses, or sentences) to create phonemes that will be converted to audio format. These phonemes are then synthesized as audio by applying a voice, which will determine parameters such as pitch and timbre; and generating an audio wave form that can be output to a speaker or written to a file.</p>
<p>You can use the output of speech synthesis for many purposes, including:</p>
<ul>
<li>Generating spoken responses to user input.</li>
<li>Creating voice menus for telephone systems.</li>
<li>Reading email or text messages aloud in hands-free scenarios.</li>
<li>Broadcasting announcements in public locations, such as railway stations or airports.</li>
</ul>
<h3 id="get-started-with-speech-on-azure">Get started with speech on Azure</h3>
<p>Microsoft Azure offers both speech recognition and speech synthesis capabilities through the <strong>Speech</strong> cognitive service, which includes the following application programming interfaces (APIs):</p>
<ul>
<li>The <strong>Speech-to-Text</strong> API</li>
<li>The <strong>Text-to-Speech</strong> API</li>
</ul>
<h4 id="the-speech-to-text-api">The speech-to-text API</h4>
<p>You can use the speech-to-text API to perform real-time or batch transcription of audio into a text format. The audio source for transcription can be a real-time audio stream from a microphone or an audio file.</p>
<p>The model that is used by the speech-to-text API, is based on the Universal Language Model that was trained by Microsoft. The data for the model is Microsoft-owned and deployed to Microsoft Azure. The model is optimized for two scenarios, conversational and dictation. You can also create and train your own custom models including acoustics, language, and pronunciation if the pre-built models from Microsoft do not provide what you need.</p>
<h5 id="real-time-transcription">Real-time transcription</h5>
<p>Real-time speech-to-text allows you to transcribe text in audio streams. You can use real-time transcription for presentations, demos, or any other scenario where a person is speaking.</p>
<h5 id="batch-transcription">Batch transcription</h5>
<p>Not all speech-to-text scenarios are real time. You may have audio recordings stored on a file share, a remote server, or even on Azure storage. You can point to audio files with a shared access signature (SAS) URI and asynchronously receive transcription results. Batch transcription should be run in an asynchronous manner because the batch jobs are scheduled on a <em>best-effort basis</em>.</p>
<h4 id="the-text-to-speech-api">The text-to-speech API</h4>
<p>The text-to-speech API enables you to convert text input to audible speech, which can either be played directly through a computer speaker or written to an audio file.</p>
<h5 id="speech-synthesis-voices">Speech synthesis voices</h5>
<p>When you use the text-to-speech API, you can specify the voice to be used to vocalize the text. The service includes multiple pre-defined voices with support for multiple languages and regional pronunciation, including <em>standard</em> voices as well as <em>neural</em> voices that leverage <em>neural networks</em> to overcome common limitations in speech synthesis with regard to intonation, resulting in a more natural sounding voice. You can also develop custom voices and use them with the text-to-speech API</p>
<h2 id="translate-text-and-speech">Translate text and speech</h2>
<h3 id="introduction-1">Introduction</h3>
<p>Artificial Intelligence (AI) can help simplify communication by translating text or speech between languages, helping to remove barriers to communication across countries and cultures.</p>
<h4 id="literal-and-semantic-translation">Literal and semantic translation</h4>
<p>Early attempts at machine translation applied <em>literal</em> translations. A literal translation is where each word is translated to the corresponding word in the target language. This approach presents some issues. For one case, there may not be an equivalent word in the target language. Another case is where literal translation can change the meaning of the phrase or not get the context correct.<br>
Artificial intelligence systems must be able to understand, not only the words, but also the <em>semantic</em> context in which they are used. In this way, the service can return a more accurate translation of the input phrase or phrases. The grammar rules, formal versus informal, and colloquialisms all need to be considered.</p>
<h4 id="text-and-speech-translation">Text and speech translation</h4>
<p><em>Text translation</em> can be used to translate documents from one language to another. <em>Speech translation</em> is used to translate between spoken languages, sometimes directly (speech-to-speech translation) and sometimes by translating to an intermediary text format (speech-to-text translation).</p>
<h3 id="get-started-translation-in-azure">Get started translation in Azure</h3>
<p>Microsoft Azure provides cognitive services that support translation. Specifically, you can use the following services:</p>
<ul>
<li>The <strong>Translator Text</strong> service, which supports text-to-text translation.</li>
<li>The <strong>Speech</strong> service, which enables speech-to-text and speech-to-speech translation.</li>
</ul>
<h4 id="text-translation-with-the-translator-text-service">Text translation with the Translator Text service</h4>
<p>The Translator Text service is easy to integrate in your applications, websites, tools, and solutions. The service uses a Neural Machine Translation (NMT) model for translation, which analyzes the semantic context of the text and renders a more accurate and complete translation as a result.</p>
<h5 id="translator-text-service-language-support">Translator Text service language support</h5>
<p>The Text Translator service supports text-to-text translation between <a href="https://docs.microsoft.com/en-us/azure/cognitive-services/translator/languages">more than 60 languages</a>. When using the service, you must specify the language you are translating <em><strong>from</strong></em> and the language you are translating <em><strong>to</strong></em> using ISO 639-1 language codes, such as <em>en</em> for English, <em>fr</em> for French, and <em>zh</em> for Chinese. Alternatively, you can specify cultural variants of languages by extending the language code with the appropriate 3166-1 cultural code - for example, <em>en-US</em> for US English, <em>en-GB</em> for British English, or <em>fr-CA</em> for Canadian French.</p>
<p>When using the Text Translator service, you can specify one <em><strong>from</strong></em> language with multiple <em><strong>to</strong></em> languages, enabling you to simultaneously translate a source document into multiple languages.</p>
<p>There’s no Python SDK for this service, but you can use its REST interface to submit requests to an endpoint over HTTP, which is relatively easy to do in Python by using the <strong>requests</strong> library. The information about the text to be translated and the resulting translated text are exchanged in JSON format.</p>
<h5 id="optional-configurations">Optional Configurations</h5>
<p>The Translator Text API offers some optional configuration to help you fine-tune the results that are returned, including:</p>
<ul>
<li><strong>Profanity filtering</strong>.</li>
<li><strong>Selective translation</strong>. You can tag content so that it isn’t translated. For example, you may want to tag code, a brand name, or a word/phrase that doesn’t make sense when localized.</li>
</ul>
<h4 id="speech-translation-with-the-speech-service">Speech translation with the Speech service</h4>
<p>The Speech service includes the following application programming interfaces (APIs):</p>
<ul>
<li><strong>Speech-to-text</strong> - used to transcribe speech from an audio source to text format.</li>
<li><strong>Text-to-speech</strong> - used to generate spoken audio from a text source.</li>
<li><strong>Speech Translation</strong> - used to translate speech in one language to text or speech in another.</li>
</ul>
<h5 id="speech-service-language-support">Speech service language support</h5>
<p>As with the Translator Text service, you can specify one source language and one or more target languages to which the source should be translated. You can translate speech into <a href="https://docs.microsoft.com/en-us/azure/cognitive-services/speech-service/language-support#speech-translation">over 60 languages</a>.</p>
<p>The source language must be specified using the extended language and culture code format, such as <em>es-US</em> for American Spanish. This requirement helps ensure that the source is understood properly, allowing for localized pronunciation and linguistic idioms.</p>
<p>The target languages must be specified using a two-character language code, such as <em>en</em> for English or <em>de</em> for German.</p>
<h2 id="create-a-language-model-with-language-understanding">Create a language model with Language Understanding</h2>
<h3 id="introduction-2">Introduction</h3>
<p>In 1950, the British mathematician Alan Turing devised the <em>Imitation Game</em>, which has become known as the <em>Turing Test</em> and hypothesizes that if a dialog is natural enough, you may not know whether you’re conversing with a human or a computer.</p>
<p>To realize the aspiration of the imitation game, computers need not only to be able to accept language as input (either in text or audio format), but also to be able to interpret the semantic meaning of the input - in other words, <em>understand</em> what is being said.</p>
<p>On Microsoft Azure, language understanding is supported through the <strong>Language Understanding Intelligent Service</strong>, more commonly known as <strong>Language Understanding</strong>. To work with Language Understanding, you need to take into account three core concepts: <em>utterances</em>, <em>entities</em>, and <em>intents</em>.</p>
<h4 id="utterances">Utterances</h4>
<p>An utterance is an example of something a user might say, and which your application must interpret. For example, when using a home automation system, a user might use the following utterances:</p>
<blockquote>
<p>“<em>Switch the fan on.</em>”</p>
<p>“<em>Turn on the light.</em>”</p>
</blockquote>
<h4 id="entities">Entities</h4>
<p>An entity is an item to which an utterance refers. For example, <strong>fan</strong> and <strong>light</strong> in the following utterances:</p>
<blockquote>
<p>“<em>Switch the <em><strong>fan</strong></em> on.</em>”</p>
<p>“<em>Turn on the <em><strong>light</strong></em>.</em>”</p>
</blockquote>
<p>You can think of the <strong>fan</strong> and <strong>light</strong> entities as being specific instances of a general <strong>device</strong> entity.</p>
<h4 id="intents">Intents</h4>
<p>An intent represents the purpose, or goal, expressed in a user’s utterance. For example, for both of the previously considered utterances, the intent is to turn a device on; so in your Language Understanding application, you might define a <strong>TurnOn</strong> intent that is related to these utterances.</p>
<p>In this table there are numerous utterances used for each of the intents. The intent should be a concise way of grouping the utterance tasks. Of special interest is the <em><strong>None</strong></em> intent. You should consider always using the None intent to help handle utterances that do not map any of the utterances you have entered. The None intent is considered a fallback, and is typically used to provide a generic response to users when their requests don’t match any other intent.</p>
<h3 id="getting-started-with-language-understanding">Getting started with Language Understanding</h3>
<p>Creating a language understanding application with Language Understanding consists of two main tasks. First you must define entities, intents, and utterances with which to train the language model - referred to as <em>authoring</em> the model. Then you must publish the model so that client applications can use it for intent and entity <em>prediction</em> based on user input.</p>
<p>If you choose to create a Language Understanding resource, you will be prompted to choose <em>authoring</em>, <em>prediction</em>, or <em>both</em> - and it’s important to note that if you choose “both”, then <em><strong>two</strong></em> resources are created - one for authoring and one for prediction.</p>
<h4 id="authoring">Authoring</h4>
<p>After you’ve created an authoring resource, you can use it to author and train a Language Understanding application by defining the entities and intents that your application will predict as well as utterances for each intent that can be used to train the predictive model.</p>
<p>Language Understanding provides a comprehensive collection of prebuilt <em>domains</em> that include pre-defined intents and entities for common scenarios; which you can use as a starting point for your model. You can also create your own entities and intents.</p>
<p>utterances you define for it to create entities for them; or you can create the entities ahead of time and then map them to words in utterances as you’re creating the intents.</p>
<p>You can write code to define the elements of your model, but in most cases it’s easiest to author your model using the Language Understanding portal - a web-based interface for creating and managing Language Understanding applications.</p>
<p><a href="https://docs.microsoft.com/en-us/learn/modules/create-language-model-with-language-understanding/2-get-started">hier weiter</a><br>
<a href="https://github.com/MicrosoftLearning/mslearn-ai900">mslearn-ai900</a></p>

