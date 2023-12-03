All experiments, implementations and tests are performed in an environment that is relatively easy to reproduce. In order to have a consistent result on different machines, we use **conda** and its configuration file `environment.yml` to create and consolidate the Python runtime environment with necessary libraries. The `environment.yml` file freezes the Python version information and the versions of libraries we use. To reproduce the environment we use, just run `conda env create --name miniai --file environment.yml`. To update the existing environment run `conda env update --name miniai --file environment.yml`. You have to activate the environment with the command `conda activate miniai` every time you open a new terminal, and then launch `jupyter notebook`. The same code can be run in the **Google Colab** environment without modification.

We use PyTorch as the default library for data fetching, implementation of the optimization algorithms, neural network model creation, and their training and application. We also use libraries such as **NumPy** to illustrate some of the internal implementation details of **PyTorch**.

# Roadmap

* [CMU B.S. AI Curriculum](https://www.cs.cmu.edu/bs-in-artificial-intelligence/curriculum)

# Mathematics

## Books

* [Mathematics for Machine Learning](https://mml-book.github.io/)
* [Algebra, Topology, Differential Calculus, and
Optimization Theory for Computer Science and Machine Learning](https://www.cis.upenn.edu/~jean/gbooks/geomath.html)
* [Linear Algebra and Optimization with Applications to Machine Learning](https://www.cis.upenn.edu/~jean/gbooks/linalg.html)

## Tools

* [Anaconda](https://www.anaconda.com/download)
* [nbdev](https://github.com/fastai/nbdev)
* [nbconvert](https://github.com/jupyter/nbconvert)
* [MATLAB](https://www.mathworks.com/products/matlab.html)
* [Octave](https://www.gnu.org/software/octave/index)
* [WolframAlpha](https://www.wolframalpha.com/)

## Libraries

* [NumPy](https://numpy.org/)
* [latexify](https://github.com/google/latexify_py)

# Numerical Optimization

## Books

* [Numerical Recipes](http://numerical.recipes/)
* [Numerical Optimization](https://link.springer.com/book/10.1007/978-0-387-40065-5)
* [Convex Optimization](https://web.stanford.edu/~boyd/cvxbook/)
* Numerical Analysis

## Tools

* [NVIDIA CUDA](https://developer.nvidia.com/about-cuda)
* [GAMS](https://www.gams.com/)
* [IBM CPLEX](https://www.ibm.com/analytics/cplex-optimizer)

## Libraries

* [SciPy](https://github.com/scipy/scipy)
* [JAX](https://github.com/google/jax)
* [Autograd](https://github.com/HIPS/autograd)
* [Torch](https://github.com/torch/torch7)
* [Theano](https://github.com/Theano/Theano)
* [MPI for Python](https://github.com/mpi4py/mpi4py)

# Statistical Learning

## Books

- [ ] [An Introduction to Statistical Learning with Applications in R](https://www.statlearning.com/)
- [ ] [The Elements of Statistical Learning](https://web.stanford.edu/~hastie/ElemStatLearn/)
- [ ] [统计学习方法](https://book.douban.com/subject/33437381/)
- [ ] [The Nature of Statistical Learning Theory](https://www.springer.com/gp/book/9780387987804)
- [ ] [Statistical Learning Theory](https://www.wiley.com/en-us/Statistical+Learning+Theory-p-9780471030034)

## Tools

* [RStudio](https://github.com/rstudio/rstudio)

# Artificial Intelligence

## Courses

- [x] [Coursera: AI for Everyone](https://www.coursera.org/learn/ai-for-everyone)
- [ ] [UC Berkeley CS188: Introduction to Artificial Intelligence](https://inst.eecs.berkeley.edu/~cs188)

## Books

* [Artificial Intelligence: A Modern Approach](http://aima.cs.berkeley.edu/)

## Methods

* Fuzzy Logic
  * Fuzzy Set

# Machine Learning

## Tutorials

- [x] [Kaggle: Intro to Machine Learning](https://www.kaggle.com/learn/intro-to-machine-learning)
- [x] [Kaggle: Intermediate Machine Learning](https://www.kaggle.com/learn/intermediate-machine-learning)
- [ ] [Kaggle: Feature Engineering](https://www.kaggle.com/learn/feature-engineering)
- [ ] [Machine Learning Yearning](https://www.deeplearning.ai/machine-learning-yearning/)

## Courses

- [ ] [Coursera: Machine Learning Specialization](https://www.coursera.org/specializations/machine-learning-introduction)
- [ ] [Coursera: Machine Learning Engineering for Production (MLOps) Specialization](https://www.coursera.org/specializations/machine-learning-engineering-for-production-mlops)
- [ ] [Stanford CS229: Machine Learning](https://cs229.stanford.edu/syllabus-fall2022.html)
- [ ] [NTU by Hung-Yi Lee: Machine Learning](https://speech.ee.ntu.edu.tw/~hylee/ml/2021-spring.php)

## Books

- [ ] Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow: Concepts, Tools, and Techniques to Build Intelligent Systems
- [ ] [Pattern Recognition and Machine Learning](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/05/prml-errata-3rd-20110921.pdf)
- [ ] [Machine Learning: A Probabilistic Perspective](https://probml.github.io/pml-book/)
- [ ] [机器学习](https://book.douban.com/subject/26708119/)
- [ ] [南瓜书PumpkinBook](https://datawhalechina.github.io/pumpkin-book)
- [ ] [Probabilistic Graphical Models: Principles and Techniques](https://mitpress.mit.edu/books/probabilistic-graphical-models)
- [ ] [Foundations of Machine Learning](https://mitpress.mit.edu/9780262039406/foundations-of-machine-learning/)
- [ ] Probably Approximately Correct

## Tools

* [Google Colab](https://colab.research.google.com/)
* [AutoGluon](https://github.com/awslabs/autogluon)
* [Streamlit](https://github.com/streamlit/streamlit)

## Libraries

* [sk-learn](https://github.com/scikit-learn/scikit-learn)
* [XGBoost](https://github.com/dmlc/xgboost)

## Projects

* [Machine Learning in NumPy](https://github.com/ddbourgin/numpy-ml)

## Methods

* Supervised Learning
  * K-Nearest Neighbors
  * Decision Trees
    * Random Forest
    * [Monte-Carlo Tree Search](https://paperswithcode.com/method/monte-carlo-tree-search)
  * [SVM](https://paperswithcode.com/method/svm)
* Unsupervised Learning
  * Clustering
    * K-Means
    * Elbow Method
* Online Leaning
* [Active Learning](https://paperswithcode.com/task/active-learning)
* [Meta-Learning](https://paperswithcode.com/task/meta-learning)
* Bayesian Network
* Boltzmann Machine

# Deep Learning

## Tutorials

- [ ] [Neural Networks and Deep Learning](http://neuralnetworksanddeeplearning.com)
- [ ] [UFLDL Tutorial](http://deeplearning.stanford.edu/tutorial/)

## Courses

- [x] [Coursera: Deep Learning Specialization](https://www.coursera.org/specializations/deep-learning)
- [ ] [Stanford CS230: Deep Learning](https://cs230.stanford.edu/lecture/)
- [ ] [MIT 6.S191 Introduction to Deep Learning](http://introtodeeplearning.com/)
- [ ] [fast.ai: Practical Deep Learning for Coders](https://course.fast.ai/)
- [ ] [CMU 10-414/714: Deep Learning Systems: Algorithms and Implementation](https://dlsyscourse.org/)
- [ ] [DeepMind x UCL: The Deep Learning Lecture](https://deepmind.com/learning-resources/deep-learning-lecture-series-2020)

## Books

- [ ] Deep Learning with Python
- [ ] [Deep Learning](https://www.deeplearningbook.org/)
- [ ] [Dive into Deep Learning](https://d2l.ai/)
- [ ] [Understanding Deep Learning](https://udlbook.github.io/udlbook/)

## Tools

* [PlotNeuralNet](https://github.com/HarisIqbal88/PlotNeuralNet)

## Libraries

* [PyTorch](https://pytorch.org/)
* [Keras](https://keras.io)
* [TensorFlow](https://github.com/tensorflow/tensorflow)
* [Caffe](https://github.com/BVLC/caffe)
* [Caffe2](https://github.com/pytorch/pytorch/tree/main/caffe2)
* [tinygrad](https://github.com/geohot/tinygrad)
* [Apache MXNet](https://github.com/apache/incubator-mxnet)
* [Deeplearning4j](https://github.com/eclipse/deeplearning4j)
* [CNTK](https://github.com/microsoft/CNTK)

# Computer Vision

## Courses

- [ ] [Stanford CS231n: Convolutional Neural Networks for Visual Recognition](https://cs231n.github.io/)

## Books

* [Computer Vision: Algorithms and Applications](http://szeliski.org/Book/)
* [Computer Vision:  Models, Learning, and Inference](http://www.computervisionmodels.com/)

## Tools

* [OpenCV](https://opencv.org/)
* [Teachable Machine](https://teachablemachine.withgoogle.com/)

## Libraries

* [scikit-image](https://scikit-image.org/)
* [ImageAI](https://github.com/OlafenwaMoses/ImageAI)
* [OpenPose](https://github.com/CMU-Perceptual-Computing-Lab/openpose)

## Projects

* DeepNude
  * [Official DeepNude Algorithm](https://gitlab.com/vitaminac/deepnude)
  * [DreamPower](https://github.com/dreamnettech/dreampower)
  * [DeepNude Explain](https://github.com/yuanxiaosc/DeepNude-an-Image-to-Image-technology)
* https://github.com/topics/deepface
  * [Deepface](https://github.com/serengil/deepface)
  * [faceswap-GAN](https://github.com/shaoanlu/faceswap-GAN)
  * [如何实现人工智能换脸DeepFake](https://github.com/Fabsqrt/BitTiger/blob/master/ArtificialIntelligent/DeepFake/README.md)
* [tesseract.js](https://github.com/naptha/tesseract.js)
* [video2x](https://github.com/k4yt3x/video2x)
* [PaintsChainer](https://github.com/pfnet/PaintsChainer)

# Natual Language Processing

## Tutorials

- [ ] [NLP Course | For You](https://lena-voita.github.io/nlp_course.html)

## Courses

- [ ] [Stanford CS224n: Natural Language Processing with Deep Learning](http://web.stanford.edu/class/cs224n/)
- [ ] [Stanford CS224d: Deep Learning for Natural Language Processing](http://cs224d.stanford.edu/)

## Books

* [Foundations of Statistical Natural Language Processing](https://nlp.stanford.edu/fsnlp/)
* [Speech and Language Processing](https://home.cs.colorado.edu/~martin/slp.html)
* [Introduction to Information Retrieval](https://nlp.stanford.edu/IR-book/)

## Tools

* [DeepSpeech](https://github.com/mozilla/DeepSpeech)

## Libraries

* [spaCy](https://spacy.io/)
* [texthero](https://github.com/jbesomi/texthero)
* [metaseq](https://github.com/facebookresearch/metaseq)

## Projects

* [Natural Language Processing Tutorial for Deep Learning Researchers](https://github.com/graykode/nlp-tutorial)

# Reinforcement Learning

## Tutorials

- [ ] [Spinning Up in Deep RL](https://spinningup.openai.com/en/latest/)

## Courses

- [ ] [UC Berkeley CS285/294 Deep Reinforcement Learning](http://rail.eecs.berkeley.edu/deeprlcourse/)
- [ ] [DeepMind x UCL: Introduction to Reinforcement Learning with David Silver](https://deepmind.com/learning-resources/-introduction-reinforcement-learning-david-silver)
- [ ] [Stanford CS234: Reinforcement Learning](http://web.stanford.edu/class/cs234/index.html)

## Books

* [Reinforcement Learning: An Introduction](http://incompleteideas.net/book/the-book.html)
* [蘑菇书EasyRL](https://datawhalechina.github.io/easy-rl)

## Tools

* [Gym](https://github.com/openai/gym)
* [AirSim](https://github.com/microsoft/AirSim)

## Methods

* Evolutionary Algorithm
  * Genetic Algorithm
  * Neuroevolution

# Large Language Models

## Courses

- [x] [DeepLearning.AI: ChatGPT Prompt Engineering for Developers](https://www.deeplearning.ai/short-courses/chatgpt-prompt-engineering-for-developers/)
- [ ] [DeepLearning.AI: Building Systems with the ChatGPT API](https://www.deeplearning.ai/short-courses/building-systems-with-chatgpt/)
- [ ] [DeepLearning.AI: LangChain for LLM Application Development](https://www.deeplearning.ai/short-courses/langchain-for-llm-application-development/)

## Libraries

* [Hugging Face: Transformers](https://github.com/huggingface/transformers)
* [xFormers](https://github.com/facebookresearch/xformers)
* [TRL - Transformer Reinforcement Learning](https://github.com/lvwerra/trl)
* [vLLM](https://github.com/vllm-project/vllm)
* [DeepSpeed](https://github.com/microsoft/DeepSpeed)
* [Guidance](https://github.com/microsoft/guidance)
* [PEFT](https://github.com/huggingface/peft)
* [Trax](https://github.com/google/trax)
* [AlpacaFarm](https://github.com/tatsu-lab/alpaca_farm)

## Tools

* [LangFlow](https://github.com/logspace-ai/langflow)
* [EasyLM](https://github.com/young-geng/EasyLM/tree/main)

## Projects

* [llama.cpp](https://github.com/ggerganov/llama.cpp)
* [alpaca.cpp](https://github.com/antimatter15/alpaca.cpp)
* [ggml](https://github.com/ggerganov/ggml)
* [nanoGPT](https://github.com/karpathy/nanoGPT)
* [miniGPT](https://github.com/karpathy/minGPT)
* [OpenLLaMA](https://github.com/openlm-research/open_llama)
* [Lit-LLaMA](https://github.com/Lightning-AI/lit-llama)
* [FinGPT](https://github.com/AI4Finance-Foundation/FinGPT)
* [LaWGPT](https://github.com/pengxiao-song/LaWGPT)

## Resources

* [Open LLM Leaderboard](https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard)

# Papers

* [Papers With Code](https://paperswithcode.com)
* [深度学习论文精读](https://github.com/mli/paper-reading)
* [Google Scholar](https://scholar.google.com/)
* [Sci-Hub Proxy](https://sci-hub.ee/)
* [arXiv](https://arxiv.org/)
* [ProQuest](https://www.proquest.com/)
* [Read Paper](https://readpaper.com/)
* [PubPeer](https://pubpeer.com/)
* [Paper Digest](https://www.paper-digest.com/)
* [Open Knowledge Map](https://openknowledgemaps.org/)
* [Web of Science](https://mjl.clarivate.com/search-results)
* [中国知网](https://www.cnki.net/)
* [Redalyc](https://www.redalyc.org/)

## To Be Read

* [What Do We Understand About Convolutional Networks?](https://arxiv.org/abs/1803.08834)
* [Distilling the Knowledge in a Neural Network](https://arxiv.org/abs/1503.02531)
* [Video-to-Video Synthesis](https://paperswithcode.com/paper/video-to-video-synthesis)
* [Different Models](https://pytorch.org/docs/stable/torchvision/models.html)
* [Learning Internal Representations by Error Propagation](http://web.stanford.edu/class/psych209a/ReadingsByDate/02_06/PDPVolIChapter8.pdf)
* [Neural Style](https://paperswithcode.com/paper/a-neural-algorithm-of-artistic-style)
* [StoryGAN](https://paperswithcode.com/paper/storygan-a-sequential-conditional-gan-for)
* [Depth-Aware Video Frame Interpolation](https://github.com/baowenbo/DAIN)
* [Pseudo-LiDAR](https://github.com/mileyan/pseudo_lidar)

# Conference

* Conference on Neural Information Processing Systems (NeurIPS)
* Computer Vision and Pattern Recognition Conference (CVPR)

# Datasets

* [Kaggle: Datasets](https://www.kaggle.com/datasets)
* [Papers With Code: Datasets](https://paperswithcode.com/datasets)
* [Google Dataset Search](https://datasetsearch.research.google.com/)
* [Hugging Face: Datasets](https://huggingface.co/datasets)

# Open Source Models

* [Kaggle: Models](https://www.kaggle.com/models)
* [Hugging Face: Models](https://huggingface.co/models)
* [Pytorch Hub](https://pytorch.org/hub/)
* [Tensorflow Hub](https://tfhub.dev/)
* [TensorFlow Model Garden](https://github.com/tensorflow/models)
