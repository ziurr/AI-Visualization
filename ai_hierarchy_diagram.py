import graphviz

# Create a Digraph object
dot = graphviz.Digraph('AI_Hierarchy', comment='AI Hierarchy Diagram', format='png')

# Set graph attributes
dot.attr(rankdir='TB', size='8,10')  # You can adjust 'rankdir' to 'LR' for a horizontal layout

# Define a recursive function to add nodes and edges
def add_nodes_edges(parent_name, child_dict):
    for child_name, grandchild in child_dict.items():
        # Add child node with custom attributes if needed
        dot.node(child_name, shape='box', style='filled', fillcolor='lightblue')
        # Add edge from parent to child
        dot.edge(parent_name, child_name)
        # If the child has its own children, recurse
        if isinstance(grandchild, dict):
            add_nodes_edges(child_name, grandchild)
        elif isinstance(grandchild, list):
            for item in grandchild:
                # Add leaf nodes
                dot.node(item, shape='ellipse')
                dot.edge(child_name, item)

# Define the AI hierarchy as a nested dictionary
ai_hierarchy = {
    'Artificial Intelligence (AI)': {
        'Machine Learning (ML)': {
            'Supervised Learning': {
                'Regression': [
                    'Linear Regression',
                    'Polynomial Regression',
                    'Support Vector Regression (SVR)',
                    'Decision Trees for Regression',
                    'Random Forest Regression',
                    'Gradient Boosting Regression',
                    'Neural Network Regression'
                ],
                'Classification': [
                    'Logistic Regression',
                    'K-Nearest Neighbors (KNN)',
                    'Support Vector Machines (SVM)',
                    'Decision Trees for Classification',
                    'Random Forest Classification',
                    'Gradient Boosting Classification',
                    'Naive Bayes Classifiers',
                    'Neural Networks for Classification',
                    'Ensemble Methods'
                ]
            },
            'Unsupervised Learning': {
                'Clustering': [
                    'K-Means Clustering',
                    'Hierarchical Clustering',
                    'DBSCAN',
                    'Mean Shift Clustering',
                    'Gaussian Mixture Models (GMM)'
                ],
                'Dimensionality Reduction': [
                    'Principal Component Analysis (PCA)',
                    'Independent Component Analysis (ICA)',
                    't-Distributed Stochastic Neighbor Embedding (t-SNE)',
                    'Uniform Manifold Approximation and Projection (UMAP)',
                    'Autoencoders'
                ],
                'Anomaly Detection': [
                    'One-Class SVM',
                    'Isolation Forest',
                    'Local Outlier Factor (LOF)'
                ]
            },
            'Semi-Supervised Learning': [
                'Self-Training Models',
                'Co-Training Methods',
                'Graph-Based Semi-Supervised Learning'
            ],
            'Reinforcement Learning (RL)': {
                'Model-Free RL': {
                    'Value-Based Methods': [
                        'Q-Learning',
                        'Deep Q-Networks (DQN)',
                        'Double DQN',
                        'Dueling DQN',
                        'Rainbow DQN'
                    ],
                    'Policy-Based Methods': [
                        'Policy Gradients',
                        'REINFORCE Algorithm',
                        'Actor-Critic Methods',
                        'Advantage Actor-Critic (A2C)',
                        'Asynchronous Advantage Actor-Critic (A3C)',
                        'Proximal Policy Optimization (PPO)',
                        'Deep Deterministic Policy Gradient (DDPG)',
                        'Twin Delayed DDPG (TD3)',
                        'Soft Actor-Critic (SAC)'
                    ]
                },
                'Model-Based RL': [
                    'Dyna-Q',
                    'Monte Carlo Tree Search (MCTS)',
                    'World Models'
                ],
                'Multi-Agent RL': [],
                'Hierarchical RL': [],
                'Inverse Reinforcement Learning (IRL)': []
            },
            'Imitation Learning': [
                'Behavior Cloning',
                'Inverse Reinforcement Learning (IRL)',
                'Apprenticeship Learning',
                'Generative Adversarial Imitation Learning (GAIL)',
                'Dataset Aggregation (DAgger)'
            ],
            'Deep Learning': {
                'Artificial Neural Networks (ANNs)': {
                    'Feedforward Neural Networks (FFNNs)': [
                        'Multilayer Perceptrons (MLPs)'
                    ],
                    'Convolutional Neural Networks (CNNs)': [
                        'LeNet',
                        'AlexNet',
                        'VGGNet',
                        'GoogLeNet (Inception Networks)',
                        'ResNet (Residual Networks)',
                        'DenseNet',
                        'EfficientNet',
                        'MobileNet',
                        'NASNet'
                    ],
                    'Recurrent Neural Networks (RNNs)': [
                        'Simple RNNs',
                        'Long Short-Term Memory Networks (LSTMs)',
                        'Gated Recurrent Units (GRUs)',
                        'Bidirectional RNNs'
                    ],
                    'Attention Mechanisms': [
                        'Additive Attention',
                        'Multiplicative Attention',
                        'Self-Attention'
                    ],
                    'Transformers': {
                        'Natural Language Processing Transformers': [
                            'Original Transformer Architecture',
                            'BERT',
                            'GPT Series',
                            'T5',
                            'XLNet',
                            'ELECTRA'
                        ],
                        'Vision Transformers (ViT)': [
                            'ViT',
                            'DeiT',
                            'Swin Transformer'
                        ],
                        'Multimodal Transformers': [
                            'VisualBERT',
                            'LXMERT',
                            'CLIP'
                        ]
                    },
                    'Autoencoders': [
                        'Basic Autoencoders',
                        'Denoising Autoencoders',
                        'Variational Autoencoders (VAEs)',
                        'Conditional VAEs (CVAEs)'
                    ],
                    'Generative Adversarial Networks (GANs)': [
                        'Basic GANs',
                        'Deep Convolutional GANs (DCGANs)',
                        'Conditional GANs (cGANs)',
                        'CycleGAN',
                        'StyleGAN',
                        'BigGAN',
                        'Wasserstein GAN (WGAN)'
                    ],
                    'Diffusion Models': [
                        'Denoising Diffusion Probabilistic Models (DDPMs)',
                        'DALL·E 2',
                        'Stable Diffusion'
                    ],
                    'Graph Neural Networks (GNNs)': [
                        'Graph Convolutional Networks (GCNs)',
                        'Graph Attention Networks (GATs)',
                        'GraphSAGE',
                        'Message Passing Neural Networks (MPNNs)'
                    ],
                    'Self-Supervised Learning': [
                        'Contrastive Learning',
                        'Masked Autoencoders',
                        'Predictive Coding'
                    ],
                    'Federated Learning': [
                        'Federated Averaging (FedAvg)',
                        'Secure Aggregation Protocols',
                        'Differential Privacy Techniques'
                    ],
                    'Multimodal Learning': [
                        'Text and Image Integration',
                        'Audio-Visual Speech Recognition',
                        'Video Captioning'
                    ],
                    'Continual Learning': [
                        'Elastic Weight Consolidation (EWC)',
                        'Synaptic Intelligence (SI)',
                        'Memory Replay Methods'
                    ],
                    'Neuro-Symbolic AI': [
                        'Logic Tensor Networks',
                        'Neural Theorem Provers',
                        'Neural Logic Machines'
                    ],
                    'Causal Learning': [
                        'Causal Discovery',
                        'Structural Causal Models',
                        'Do-Calculus'
                    ],
                    'Meta-Learning': [
                        'Model-Agnostic Meta-Learning (MAML)',
                        'Prototypical Networks',
                        'Relation Networks'
                    ],
                    'Few-Shot and Zero-Shot Learning': [
                        'Siamese Networks',
                        'Matching Networks',
                        'GPT-3 Zero-Shot Capabilities'
                    ],
                    'Large Language Models (LLMs)': [
                        'GPT-3',
                        'GPT-4',
                        'T5',
                        'Megatron-LM',
                        'BLOOM'
                    ],
                    'Foundation Models': [
                        'GPT-3',
                        'CLIP',
                        'DALL·E',
                        'Flamingo'
                    ]
                }
            },
            'Probabilistic Models': [
                'Bayesian Networks',
                'Markov Random Fields',
                'Hidden Markov Models (HMMs)',
                'Conditional Random Fields (CRFs)'
            ],
            'Evolutionary Algorithms': [
                'Genetic Algorithms',
                'Genetic Programming',
                'Evolution Strategies',
                'Differential Evolution'
            ],
            'Swarm Intelligence': [
                'Particle Swarm Optimization',
                'Ant Colony Optimization'
            ],
            'Optimization Algorithms': [
                'Gradient Descent',
                'Stochastic Gradient Descent (SGD)',
                'Adam Optimizer',
                'RMSprop',
                'Adagrad',
                'Adadelta'
            ]
        },
        'Natural Language Processing (NLP)': {
            'Tasks': [
                'Language Modeling',
                'Machine Translation',
                'Sentiment Analysis',
                'Named Entity Recognition (NER)',
                'Part-of-Speech Tagging',
                'Question Answering',
                'Text Summarization',
                'Dialogue Systems',
                'Text Classification',
                'Natural Language Generation',
                'Semantic Parsing',
                'Coreference Resolution'
            ],
            'Techniques': [
                'Bag-of-Words Models',
                'N-gram Models',
                'Word Embeddings',
                'Recurrent Neural Networks',
                'Transformers'
            ],
            'Pretrained Language Models': [
                'BERT',
                'RoBERTa',
                'ALBERT',
                'GPT Series',
                'T5',
                'XLNet',
                'ELECTRA',
                'Megatron-LM',
                'BLOOM'
            ],
            'Conversational AI': [
                'Open-Domain Chatbots',
                'Task-Oriented Dialogue Systems',
                'Transfer Learning in Dialogue Systems'
            ],
            'Multimodal NLP': [
                'CLIP',
                'VisualBERT',
                'LXMERT'
            ],
            'Cross-Lingual Models': [
                'XLM-R',
                'mBERT'
            ]
        },
        'Computer Vision': {
            'Tasks': [
                'Image Classification',
                'Object Detection',
                'Image Segmentation',
                'Facial Recognition',
                'Activity Recognition',
                'Image Captioning',
                'Video Analysis',
                '3D Reconstruction',
                'Depth Estimation',
                'Pose Estimation'
            ],
            'Techniques': [
                'Convolutional Neural Networks',
                'Vision Transformers (ViT)',
                'Generative Models',
                'Self-Supervised Learning',
                '3D Computer Vision',
                'Object Detection Models',
                'Image Segmentation Models'
            ],
            'Applications': [
                'Autonomous Vehicles',
                'Medical Image Analysis',
                'Augmented Reality (AR)',
                'Remote Sensing'
            ]
        },
        'Robotics and Autonomous Systems': {
            'Perception Systems': [
                'Sensor Fusion',
                'SLAM',
                'Object Recognition'
            ],
            'Motion Planning and Control': [
                'Path Planning Algorithms',
                'Control Systems'
            ],
            'Manipulation and Grasping': [
                'Robotic Arm Control',
                'Grasp Planning',
                'Tactile Sensing'
            ],
            'Learning in Robotics': [
                'Reinforcement Learning',
                'Imitation Learning',
                'Sim-to-Real Transfer Learning'
            ],
            'Human-Robot Interaction': [
                'Social Robotics',
                'Natural Language Interaction',
                'Gesture Recognition',
                'Assistive Robotics'
            ],
            'Robotic Navigation': [
                'Obstacle Avoidance',
                'Autonomous Exploration'
            ],
            'Applications': [
                'Industrial Automation',
                'Service Robots',
                'Medical Robotics',
                'Space Robotics'
            ]
        },
        'Speech and Audio Processing': {
            'Automatic Speech Recognition (ASR)': [
                'DeepSpeech',
                'Transformer ASR Models'
            ],
            'Text-to-Speech (TTS)': [
                'WaveNet',
                'Tacotron',
                'FastSpeech',
                'Neural Voice Cloning'
            ],
            'Speaker Identification and Verification': [
                'Voice Biometrics',
                'Speaker Diarization'
            ],
            'Emotion Recognition': [
                'Prosody Analysis',
                'Affective Computing'
            ],
            'Audio Classification': [
                'Music Genre Classification',
                'Environmental Sound Classification'
            ],
            'Speech Enhancement': [
                'Noise Reduction',
                'Speech Dereverberation'
            ],
            'Multilingual Speech Models': [
                'Multilingual ASR',
                'Code-Switching Recognition'
            ]
        },
        'Recommender Systems': {
            'Collaborative Filtering': [
                'User-Based Filtering',
                'Item-Based Filtering',
                'Matrix Factorization',
                'Singular Value Decomposition (SVD)'
            ],
            'Content-Based Filtering': [
                'Feature Extraction',
                'Similarity Measures'
            ],
            'Hybrid Methods': [
                'Combining Collaborative and Content-Based Approaches'
            ],
            'Deep Learning for Recommendations': [
                'Neural Collaborative Filtering',
                'Autoencoders for Recommendations',
                'Graph-Based Recommendations'
            ],
            'Context-Aware Recommendations': [
                'Location-Based Services',
                'Time-Aware Recommendations'
            ],
            'Sequential and Session-Based Recommendations': [
                'Recurrent Neural Networks',
                'Transformers'
            ],
            'Privacy-Preserving Recommendations': [
                'Differential Privacy',
                'Federated Learning'
            ]
        },
        'Meta-Learning and AutoML': {
            'Meta-Learning': [
                'Model-Agnostic Meta-Learning (MAML)',
                'Meta-SGD',
                'Reptile Algorithm',
                'Prototypical Networks',
                'Relation Networks',
                'Matching Networks'
            ],
            'Automated Machine Learning (AutoML)': [
                'Hyperparameter Optimization',
                'Neural Architecture Search (NAS)'
            ],
            'Transfer Learning': [
                'Fine-Tuning Pretrained Models',
                'Domain Adaptation',
                'Multi-Task Learning'
            ]
        },
        'Explainable AI (XAI) and Fairness': {
            'Interpretable Models': [
                'Decision Trees',
                'Rule-Based Models',
                'Generalized Additive Models (GAMs)',
                'Explainable Boosting Machines (EBMs)'
            ],
            'Post-Hoc Explanation Methods': [
                'Feature Importance',
                'LIME',
                'SHAP',
                'Counterfactual Explanations',
                'Saliency Maps'
            ],
            'Causal Explainability': [
                'Structural Causal Models',
                'Do-Calculus'
            ],
            'Fairness Metrics and Bias Mitigation': [
                'Demographic Parity',
                'Equal Opportunity',
                'Adversarial Debiasing'
            ],
            'AI Safety and Alignment': [
                'Robustness to Adversarial Attacks',
                'Safe Reinforcement Learning',
                'Value Alignment',
                'AI Governance and Policy'
            ],
            'Ethical Frameworks': [
                'Transparency',
                'Accountability',
                'Privacy Considerations'
            ]
        },
        'Optimization and Search Algorithms': {
            'Gradient-Based Optimization': [
                'Gradient Descent',
                'Stochastic Gradient Descent (SGD)',
                'Momentum Methods',
                'Adaptive Learning Rate Methods'
            ],
            'Evolutionary Algorithms': [
                'Genetic Algorithms',
                'Evolution Strategies',
                'Genetic Programming'
            ],
            'Swarm Intelligence': [
                'Particle Swarm Optimization',
                'Ant Colony Optimization'
            ],
            'Bayesian Optimization': [
                'Gaussian Processes',
                'Tree-structured Parzen Estimators (TPE)'
            ],
            'Simulated Annealing': [],
            'Search Algorithms in AI': [
                'Breadth-First Search (BFS)',
                'Depth-First Search (DFS)',
                'A* Search',
                'Dijkstra\'s Algorithm',
                'Minimax Algorithm',
                'Alpha-Beta Pruning',
                'Monte Carlo Tree Search (MCTS)'
            ]
        },
        'Edge AI and Embedded Systems': {
            'TinyML': [
                'Model Compression Techniques',
                'Quantization',
                'Pruning',
                'Knowledge Distillation',
                'On-Device Machine Learning'
            ],
            'Hardware for AI': [
                'GPUs',
                'TPUs',
                'FPGAs',
                'ASICs'
            ],
            'Applications': [
                'IoT Devices',
                'Mobile AI Applications',
                'Real-Time Inference'
            ]
        },
        'Federated Learning and Privacy-Preserving AI': {
            'Federated Learning Frameworks': [
                'Federated Averaging (FedAvg)',
                'Secure Aggregation Protocols',
                'Differential Privacy Techniques'
            ],
            'Split Learning': [],
            'Encrypted Computation': [
                'Homomorphic Encryption',
                'Secure Multi-Party Computation'
            ],
            'Applications': [
                'Healthcare Data Analysis',
                'Collaborative Learning Across Institutions'
            ]
        },
        'Multimodal Learning': {
            'Models Processing Multiple Data Types': [
                'Text and Image Integration',
                'Audio-Visual Speech Recognition',
                'Video Captioning'
            ],
            'Multimodal Transformers': [
                'CLIP',
                'VisualBERT',
                'LXMERT',
                'UNITER'
            ],
            'Applications': [
                'Autonomous Driving',
                'Robotics',
                'Healthcare Diagnostics'
            ]
        },
        'Causal Inference and Causal Discovery': {
            'Causal Graphical Models': [],
            'Structural Equation Modeling': [],
            'Do-Calculus': [],
            'Counterfactual Reasoning': [],
            'Applications': [
                'Epidemiology',
                'Economics',
                'Social Sciences'
            ]
        },
        'Neuro-Symbolic AI': {
            'Combining Neural Networks with Symbolic Reasoning': [
                'Logic Tensor Networks',
                'Neural Theorem Provers',
                'Neural Logic Machines'
            ],
            'Applications': [
                'Knowledge Graph Reasoning',
                'Program Synthesis',
                'Commonsense Reasoning'
            ]
        },
        'Continual Learning (Lifelong Learning)': {
            'Techniques to Learn Sequentially': [
                'Elastic Weight Consolidation (EWC)',
                'Synaptic Intelligence (SI)',
                'Memory Replay Methods',
                'Progressive Neural Networks',
                'Dynamically Expandable Networks'
            ],
            'Applications': [
                'Personalization',
                'Robotics',
                'Adaptive Systems'
            ]
        },
        'Self-Supervised Learning': {
            'Learning Representations Without Labels': [
                'Contrastive Learning',
                'Masked Autoencoders',
                'Predictive Coding'
            ],
            'Applications': [
                'Pretraining Models',
                'Representation Learning'
            ]
        },
        'Foundation Models': {
            'Large-Scale Pretrained Models Adaptable to Many Tasks': [
                'GPT-3',
                'GPT-4',
                'T5',
                'CLIP',
                'DALL·E',
                'Flamingo',
                'Gopher'
            ],
            'Properties': [
                'Few-Shot Learning',
                'Zero-Shot Learning',
                'Multitask Learning'
            ],
            'Applications': [
                'Language Understanding and Generation',
                'Image Generation',
                'Code Generation'
            ]
        },
        'Ethics, Fairness, and AI Alignment': {
            'AI Safety Research': [
                'Robustness',
                'Interpretability',
                'Prevention of Unintended Behaviors'
            ],
            'Fairness Metrics': [
                'Statistical Parity',
                'Equalized Odds'
            ],
            'Bias Mitigation Techniques': [
                'Preprocessing Methods',
                'In-Processing Methods',
                'Post-Processing Methods'
            ],
            'AI Governance and Policy': [
                'Regulatory Frameworks',
                'Ethical Guidelines'
            ],
            'AI and Society': [
                'Impact on Employment',
                'Privacy Concerns'
            ]
        },
        'AI for Science and Medicine': {
            'Protein Folding': [
                'AlphaFold'
            ],
            'Drug Discovery': [
                'Virtual Screening',
                'Molecular Property Prediction'
            ],
            'Climate Modeling': [
                'Weather Prediction',
                'Environmental Monitoring'
            ],
            'Materials Science': [
                'Materials Design',
                'Simulation Acceleration'
            ]
        },
        'Human-Centered AI': {
            'User Experience (UX) in AI Systems': [],
            'Trust and Transparency': [],
            'Collaborative AI': [
                'Human-in-the-Loop Learning',
                'Interactive Machine Learning'
            ],
            'AI for Accessibility': [
                'Assistive Technologies',
                'Adaptive Interfaces'
            ]
        },
        'Quantum Machine Learning': {
            'Quantum Computing Principles': [],
            'Quantum Algorithms for ML': [
                'Quantum Neural Networks',
                'Quantum Support Vector Machines',
                'Variational Quantum Circuits'
            ],
            'Applications': [
                'Optimization Problems',
                'Cryptography',
                'Material Simulation'
            ]
        },
        'Cognitive Computing and Affective Computing': {
            'Cognitive Computing': [
                'IBM Watson',
                'Knowledge Representation'
            ],
            'Affective Computing': [
                'Emotion Recognition',
                'Sentiment Analysis',
                'Human-Computer Interaction'
            ]
        }
    }
}

# Start building the graph from the root node
root_node = 'Artificial Intelligence (AI)'
dot.node(root_node, shape='box', style='filled', fillcolor='lightgrey')

dot.attr(rankdir='LR', size='1000,1000')
add_nodes_edges(root_node, ai_hierarchy[root_node])

# Render the graph to a file
dot.render('ai_hierarchy_diagram', view=True)
