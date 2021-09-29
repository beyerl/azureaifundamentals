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
<p>Microsoft Azure provides the following cognitive services to help you create computer vision solutions:</p>
<p>Computer vision services in Microsoft Azure</p>
<p>Service</p>
<p>Capabilities</p>
<p><strong>Computer Vision</strong></p>
<p>You can use this service to analyze images and video, and extract descriptions, tags, objects, and text.</p>
<p><strong>Custom Vision</strong></p>
<p>Use this service to train custom image classification and object detection models using your own images.</p>
<p><strong>Face</strong></p>
<p>The Face service enables you to build face detection and facial recognition solutions.</p>
<p><strong>Form Recognizer</strong></p>
<p>Use this service to extract information from scanned forms and invoices.</p>
<p>(hier weiter)[<a href="https://docs.microsoft.com/en-us/learn/modules/get-started-ai-fundamentals/5-understand-natural-language-process">https://docs.microsoft.com/en-us/learn/modules/get-started-ai-fundamentals/5-understand-natural-language-process</a>]</p>

