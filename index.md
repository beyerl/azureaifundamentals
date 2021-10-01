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
<p><a href="https://docs.microsoft.com/en-us/learn/modules/create-regression-model-azure-machine-learning-designer/inference-pipeline">hier weiter</a></p>

