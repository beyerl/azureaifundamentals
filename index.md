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
<p><a href="https://docs.microsoft.com/en-us/learn/modules/use-automated-machine-learning/create-workspace">Hier weiter</a></p>

