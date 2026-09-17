## Below are the papers reviewed by the team to support each step and the progress made.

Pinheiro et al. (2025) systematically evaluated twelve feature scaling techniques across fourteen machine learning algorithms and sixteen datasets. Their results show that ensemble methods such as Random Forest, XGBoost, CatBoost, and LightGBM remain robust regardless of scaling, while models such as Logistic Regression, SVM, TabNet, and MLP are highly sensitive to the chosen scaler. The study provides reproducible, model-specific guidance on the importance of selecting an appropriate scaling strategy during preprocessing, particularly for linear models where unscaled features of differing magnitudes can distort model learning. 

Subramanya, Bhatt and Bhansali (2023) show that the standard industry solution to the training-serving skew problem is a feature store, essentially a pre-computed lookup table that maps an entity ID to its feature values. They note that feature correctness violations related to online-offline skew and data leakage are common in production ML, and that precomputing and storing features is the recommended way to ensure consistency between training and serving. This is directly relevant to the reviewer_stats.json file generated in this notebook, which acts as a lightweight version of exactly this pattern using reviewerID as the key to retrieve historical review frequency at inference time, and falling back to the population median when a reviewer is not found.

Talaat (2023) evaluated eight hybrid BERT-based models combining DistilBERT and RoBERTa with BiGRU and BiLSTM layers for three class sentiment classification (positive, negative, neutral) across three real world datasets: Airlines, CrowdFlower, and Apple Twitter. The results showed that BERT based models consistently outperformed all seven classical machine learning classifiers tested, with the highest accuracy of 91.72% recorded on the Apple dataset. This is directly relevant to this project, which uses the same three class sentiment structure on short informal review texts. Additionally, the study's inclusion of classical machine learning models as benchmarks supports the use of Logistic Regression and SVM in this project as valid baseline classifiers against which BERT's performance is compared.

Salminen et al (2022) examine the creation and detection of fake online product reviews, focusing on how deceptive reviews can be generated and identified using computational techniques. In their study, authentic reviews from the Amazon review dataset were used as the basis for generating deceptive reviews using language models such as ULMFiT and GPT-2. The generated fake reviews were combined with genuine reviews to form a labelled dataset for analysis. The authors analyse the linguistic characteristics that distinguish authentic reviews from fabricated ones, highlighting differences in writing style, sentiment intensity and content structure. Machine learning algorithms were then applied to detect deceptive reviews by learning patterns present in the fabricated text. Their findings demonstrate that fake reviews often follow identifiable linguistic patterns that can be detected through automated methods, although increasingly sophisticated text generation techniques continue to present challenges for detection systems.

Yuan et al. (2022) conducted an empirical study examining how the size and composition of the SHAP background dataset affects explanation reliability. Their findings show that SHAP values and variable rankings fluctuate when using different background datasets acquired from random sampling, and that SHAP stability improves as the background sample size increases. The authors conclude that users should not unquestioningly trust SHAP explanations derived from small or unrepresentative background datasets, as this can produce misleading attributions for features that are genuinely predictive.

Hameed et al. (2022) argued that ablation is a practical way to validate whether a model's explanations are trustworthy. By perturbing input variables in rank order of importance, the goal is to assess the sensitivity of the model's performance, where removing important features should cause larger drops in performance than removing less important ones. This is consistent with the findings in this project, where removing review_frequency caused recall to drop to zero, confirming it as the most critical engineered feature.

Soldner et al. (2021) examine methods for detecting fake online reviews using machine learning techniques and review metadata. The study highlights that platform-specific indicators such as verified purchase information can serve as useful signals when identifying potentially deceptive reviews. By combining metadata features with textual analysis, the authors demonstrate that machine learning models can improve the detection of fraudulent reviews on online platforms 

Elmogy et al. (2021) proposed a supervised machine learning approach for detecting fake reviews on e-commerce platforms, evaluating five classifiers Logistic Regression, SVM, KNN, Naive Bayes, and Random Forest on a real Yelp restaurant review dataset using TF-IDF textual features combined with behavioural features such as writing style and reviewer frequency patterns. Their findings demonstrated that Logistic Regression achieved the highest accuracy among all classifiers tested, outperforming more complex models including Random Forest. The study further showed that incorporating behavioural features alongside textual features significantly improved detection accuracy. These findings are directly relevant to this project, as they provide experimental validation for the selection of both Logistic Regression and SVM as the primary models for suspicious review detection. The alignment between the paper's TF-IDF feature extraction approach and the methodology adopted in this project further strengthens this justification. Furthermore, the study's emphasis on behavioural and stylistic writing signals supports the rationale for also employing BERT, which unlike TF-IDF-based models is capable of encoding contextual, tonal, and stylistic features of review language through its bidirectional transformer architecture.

Hussain et al (2020) investigate methods for detecting spam reviews by combining linguistic analysis with behavioural characteristics of reviewers. The study highlights that deceptive reviews often exhibit identifiable linguistic patterns, such as exaggerated sentiment, repetitive language and unusual writing structures. In addition to textual analysis, the authors examine reviewer behaviour, including posting frequency, rating patterns and review activity, to identify suspicious accounts. By integrating both linguistic features and behavioural indicators, the research demonstrates that hybrid approaches can improve the accuracy of spam review detection systems. The study contributes to the broader literature by emphasising that analysing both the content of reviews and the behaviour of reviewers provides a more reliable strategy for identifying deceptive online reviews. 

Uchendu et al. (2020) investigate the detection of machine-generated text, including deceptive content produced by neural language models. The authors analyse linguistic characteristics of generated text and demonstrate that machine-generated reviews often exhibit identifiable stylistic patterns that differ from human-written reviews. Their work shows that computational methods can be used to distinguish synthetic text from authentic user-generated content, which is important for detecting fake reviews created using automated systems.

Zellers et al. (2019) study the detection of neural-generated text and introduce methods for identifying machine-generated content. The research demonstrates that models trained to recognise patterns in generated text can effectively distinguish synthetic content from real text. This work highlights the growing challenge posed by advanced text generation models and emphasises the need for robust detection techniques to identify deceptive or fabricated online content.

Meyes et al. (2019) introduced ablation studies as a method for understanding which parts of a model actually matter. Their work shows that by systematically removing individual components and measuring the drop in performance, it is possible to identify which features are genuinely important to a model's predictions. They argue that ablation studies are a feasible method to investigate knowledge representations in models and are especially helpful for examining which components drive performance. This supports the approach taken in this project, where features were removed one at a time to confirm that behavioural features such as review_frequency were critical to the suspicious review detection model

Devlin et al. (2018) introduced BERT, a transformer-based model that learns bidirectional context from text, allowing it to better understand the meaning of words in a sentence. The model is pre-trained on large datasets and can be fine-tuned for tasks such as sentiment analysis.

Lundberg and Lee (2017) introduced SHapley Additive exPlanations (SHAP), a unified framework grounded in cooperative game theory that assigns each feature an importance value for a given prediction. The framework identifies a class of additive feature importance measures and provides theoretical results showing there is a unique solution in this class with a set of desirable properties. SHAP has since become a widely adopted tool for post-hoc model explanation, offering both instance-level and global feature importance, particularly in complex models where internal decision logic is not directly interpretable.

Yao et al. (2017) highlight the importance of gathering real review data rather than relying only on synthetic or crowdsourced deceptive reviews. Their work demonstrates that combining behavioural signals and real platform data can improve the reliability of datasets used for training deception detection models.

Zhang, Zhao and LeCun (2015) used several large scale text classification datasets where labels were already assigned according to predefined categories. In the case of sentiment datasets, the labels were derived from review ratings, where numerical rating scores were converted into sentiment classes such as positive and negative. These labelled datasets were then used to train and evaluate the proposed character-level convolutional neural network model.

Rayana and Akoglu (2015) proposed a collective opinion spam detection method that analyses relationships between reviewers, products, and ratings. Their approach models review data as a network and detects suspicious patterns such as coordinated review behaviour or abnormal rating distributions. This method helps identify potential spam reviews without requiring extensive labelled datasets and is particularly useful for large-scale e-commerce platforms. Their work shows the importance of combining multiple feature types, which aligns with our hybrid approach.

Li et al. (2015) provided a survey of fake review detection techniques. They categorised methods into linguistic, behavioural, and network-based approaches. Their work helped guide the overall design of our system.


Sculley et al. (2015) identified training-serving skew as one of the most damaging and hard-to-detect problems in real-world machine learning systems. They argue that machine learning systems can quickly accumulate hidden maintenance costs, with training-serving skew being one of the key risk factors that can silently degrade model performance in production. In simple terms, this happens when a feature is available during training but cannot be computed in the same way when the model is actually being used. In this project, review_frequency was computed from the full dataset during training, but at inference time a new review comes in with no historical context, meaning the model would have no way to calculate it. The reviewer_stats.json lookup table was built specifically to solve this, by storing each reviewer's historical frequency so it can be retrieved at serving time.


Hutto and Gilbert (2014) introduce VADER (Valence Aware Dictionary and sEntiment Reasoner), a rule-based sentiment analysis model designed for social media text. The model combines a sentiment lexicon with heuristics that account for linguistic features such as punctuation, capitalisation, emoticons, abbreviations and degree modifiers commonly found in online communication. Unlike machine learning approaches that require large labelled datasets, VADER relies on a predefined dictionary to determine sentiment polarity and intensity. While the model performs effectively on informal online text, rule based approaches remain limited in handling complex linguistic phenomena such as sarcasm and contextual expressions.

Li et al. (2014) propose a semi-supervised learning approach for detecting deceptive reviews when labelled data is limited. The study shows that a small set of manually labelled reviews can be used to propagate labels to larger unlabelled datasets through machine learning techniques. This approach allows models to learn patterns associated with deceptive reviews while reducing the need for extensive manual annotation, thereby improving the scalability of spam review detection systems.

Mukherjee et al. (2013) investigate group spam behaviour in online review platforms and propose behavioural analysis techniques for detecting coordinated fake reviews. The study identifies suspicious patterns such as burst reviewing, where multiple reviews are posted within a short time frame, and groups of reviewers posting similar ratings for the same products. The authors show that analysing reviewer behaviour, rating distributions and temporal patterns can help uncover organised spam campaigns aimed at manipulating product reputation 

Ott et al. (2013) extend previous work on deceptive opinion spam by focusing on negative deceptive reviews. The study analyses linguistic differences between truthful and deceptive negative reviews and shows that deceptive reviews often contain distinctive stylistic and emotional patterns. By examining lexical features and psychological indicators within the text, the authors demonstrate that linguistic analysis can effectively contribute to identifying deceptive opinion spam in online reviews.

Fei et al. (2013) analysed time-series data to detect sudden spikes in reviews. They identified burstiness as a strong signal of spam campaigns. This supports our inclusion of burst review as a key feature.

Wang and Manning (2012) investigate  baseline methods for sentiment and topic classification. The study evaluates different text representation techniques and demonstrates that bigram features combined with logistic regression can achieve strong performance compared to more complex models. The authors show that despite the simplicity of the approach, using n-gram representations, particularly bigrams captures contextual information that improves classification accuracy in sentiment analysis tasks. Their findings highlight that wel designed baseline models can perform competitively with more sophisticated algorithms, emphasising the importance of strong baseline methods when evaluating text classification systems

Liu (2012) provides a comprehensive foundation for the field of sentiment analysis, establishing a taxonomy that distinguishes between document-level, sentence-level and aspect-level opinion mining. The author argues that sentiment analysis extends beyond simple polarity detection to include the identification of opinion holders, opinion targets and the contextual factors that influence sentiment. Liu (2012) further emphasises that word sentiment is highly domain-dependent, meaning that a term considered positive in one context may carry negative connotations in another. This observation has significantly influenced subsequent research on domain adaptation in opinion mining systems, highlighting the importance of contextual understanding in sentiment classification.

Mukherjee et al. (2012) introduced group-based spam detection by identifying coordinated review patterns. Their model used behavioural and temporal features such as burst activity and reviewer similarity. This directly relates to our burst review feature.

Ott et al. (2011) explore the detection of deceptive opinion spam using linguistic cues. The authors create a dataset of deceptive reviews through manual annotation and analyse textual characteristics associated with deception. Their findings suggest that fake reviews often contain exaggerated sentiment, generic descriptions and persuasive language intended to influence readers. The study demonstrates that linguistic analysis and machine learning techniques can be used to distinguish deceptive reviews from genuine ones based on writing style and sentiment patterns. Their findings demonstrate that text-based features can effectively detect fake reviews, which supports our use of sentiment mismatch.

Lim et al. (2010) proposed detecting fake reviewers based on abnormal rating behaviours. They analysed features such as rating deviation, review frequency, and extreme ratings. Their work shows that user behaviour is a strong indicator of spam, which is reflected in our use of features like review frequency and extreme ratings.

Jindal and Liu (2008) examine the problem of opinion spam in online review systems and propose methods for identifying deceptive reviews using textual and behavioural patterns. The authors highlight that fake reviews often appear as duplicate or near duplicate content, where the same or highly similar reviews are posted multiple times across products. They also identify unusual rating patterns and suspicious reviewer behaviour as important indicators of spam reviews. Their work demonstrates that analysing both the content of reviews and reviewer activity can help detect deceptive opinion spam in online platforms. This paper highlights the importance of combining linguistic and behavioural features, which influenced our feature selection.

Pang and Lee (2008) provide a comprehensive survey of sentiment analysis research, tracing the development of the field from early lexicon-based approaches to supervised machine learning methods. Their work highlights the significant impact of domain specificity on classifier performance, demonstrating that models trained on one genre of text often perform poorly when applied to another. Pang and Lee (2008) also identify subjectivity detection as an important preprocessing step prior to polarity classification, proposing a two stage pipeline that has since become widely adopted in sentiment analysis research. In addition, their evaluation of linguistic features such as n-grams and part-of-speech tags remains an important reference for feature engineering in text-based sentiment classification tasks.

Mitchell (1997) provides a foundational overview of machine learning algorithms and principles, including probabilistic models such as Naive Bayes. The book explains how machine learning methods can learn patterns from data to perform classification tasks, making it a key reference for understanding algorithms used in text classification and sentiment analysis.














## References

Devlin, J., Chang, M., Lee, K. and Toutanova, K. (2019) Bert: Pre-training of Deep Bidirectional Transformers For Language Understanding. Proceedings of the 2019 Conference of the North American Chapter of the Association For Computational Linguistics: Human Language Technologies [online]. 1, pp. 4171-4186. [Accessed 05 April 2026].

Elmogy, A.M., Tariq, U., Mohammed, A. and Ibrahim, A. (2021) Fake Reviews Detection Using Supervised Machine Learning. International Journal of Advanced Computer Science and Applications [online]. 1 (12), pp. 601-606. [Accessed 02 April 2026].

Fei, G., Mukherjee, A., Liu, B., Hsu, M., Castellanos, M. and Ghosh, R. (2013) Exploiting Burstiness in Reviews For Review Spammer Detection. Proceedings of the International Aaai Conference on Web and Social Media [online]. 1 (7), pp. 175-184. [Accessed 10 April 2026].

Hameed, I., Sharpe, S., Barcklow, D., Au-yeung, J., Verma, S., Huang, J., Barr, B. and Bruss, C.B. (2022) . Based-xai: Breaking Ablation Studies Down For Explainable Artificial Intelligence [online]. [Accessed 15 April 2026].

Hussain, N.H., Mirza, H.T., Hussain, I.H., Iqbal, F.I. and Memon, I.M. (2020) Spam Review Detection Using the Linguistic and Spammer Behavioral Methods. Ieee Access [online]. 8, p. 53801–53816. [Accessed 20 March 2026].

Hutto, C.J. and Gilbert, E. (2014) Vader: A Parsimonious Rule-based Model For Sentiment Analysis of Social Media Text. Proceedings of the International Aaai Conference on Web and Social Media [online]. 8 (1), pp. 216-225. [Accessed 05 March 2026].

Jindal,, N. and Liu, B. (2008) Opinion Spam and Analysis. Proceedings of the 2008 International Conference on Web Search and Data Mining (Wsdm '08) [online]., pp. 219-230. [Accessed 11 March 2026].

Lundberg, S.M. and Lee, S.I. (2017) 'A unified approach to interpreting model predictions', in Advances in Neural Information Processing Systems 30. [Online] Red Hook, NY: Curran Associates, pp. 4765–4774. Available from: neurips.cc [Accessed 15 April 2026].

Liu, B. (2012) Sentiment Analysis and Opinion Mining. 1st ed. California: Morgan and Claypool Publishers.

Li, J., Cardie, C. and Li, S. (2013) Topicspam: A Topic-model Based Approach For Spam Detection. Proceedings of the 51st Annual Meeting of the Association For Computational Linguistics [online]. 2, pp. 217-221. [Accessed 20 March 2026].

Li, J., Ott, M., Cardie, C. and Hovy, E. (2014) Towards a General Rule For Identifying Deceptive Opinion Spam. Proceedings of the 52nd Annual Meeting of the Association For Computational Linguistics [online]. 1, pp. 1566-1576. [Accessed 25 March 2026].


Lim, E.P., Nguyen, V.A., Jindal, N., Liu, B. and Lauw, H.W. (2010) Detecting Product Review Spammers Using Rating Behaviors. Proceedings of the 19th Acm International Conference on Information and Knowledge Management [online]., pp. 939-948. [Accessed 09 April 2026].


Meyes, R., Lu, M., de Puiseau, C.W. and Meisen, T. (2019) Ablation studies in artificial neural networks. [Preprint] Available from: https://arxiv.org/abs/1901.08644 [Accessed 15 April 2026].

Mitchell, T.M. (1997) Machine Learning. 1st ed. New York: McGraw-hill.

Mukherjee, A., Liu, B. and Glance, N. (2012) 'Spotting fake reviewer groups in consumer reviews', in Proceedings of the 21st International Conference on World Wide Web. [Online] pp. 191–200. Available from: https://doi.org/10.1145/2187836.2187863 [Accessed 18 April 2026].

Mukherjee, A., Venkataraman, V., Liu, B. and Glance, N. (2013) 'What Yelp fake review filter might be doing?', in Proceedings of the 7th International AAAI Conference on Web and Social Media. [Online] pp. 409–418. Available from: https://ojs.aaai.org/index.php/ICWSM/article/view/14389 [Accessed 18 April 2026].

Ott, M., Choi, Y., Cardie, C. and Hancock, J.T. (2011) 'Finding deceptive opinion spam by any stretch of the imagination', in Proceedings of the 49th Annual Meeting of the Association for Computational Linguistics: Human Language Technologies. [Online] pp. 309–319. Available from: https://aclanthology.org/P11-1032 [Accessed 19 April 2026].

Ott, M., Cardie, C. and Hancock, J.T. (2013) 'Negative deceptive opinion spam', in Proceedings of the 2013 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies. [Online] pp. 497–501. Available from: aclanthology.org [Accessed 28 April 2026]

Pang,, B. and Lee, L. (2008) Opinion Mining and Sentiment Analysis. Foundations and Trends in Information Retrieval [online]. 2 (2), pp. 1-135. [Accessed 05 March 2026].

Pinheiro, J.M.H., Cavalcanti, G.D.C. and Ren, T.I. (2025) The impact of feature scaling in machine learning: effects on regression and classification tasks. [Preprint] Available from: https://arxiv.org/abs/2506.08274 [Accessed 26 April 2026].

Rayana, S. and Akoglu, L. (2015) 'Collective opinion spam detection: bridging review networks and metadata', in Proceedings of the 21st ACM SIGKDD International Conference on Knowledge Discovery and Data Mining. [Online] pp. 985–994. Available from: https://doi.org/10.1145/2783258.2783370 [Accessed 8 April 2026].

Salminen, J., Kandpal,, C., Kamel, A.M., Jung, S., Jansen, B.J. and , (2022) Creating and Detecting Fake Reviews of Online Products. Journal of Retailing and Consumer Services [online]. 64 [Accessed 30 March 2026].

Sculley, D., Holt, G., Golovin, D., Davydov, E., Phillips, T., Ebner, D., Chaudhary, V., Young, M., Crespo, J. and Dennison, D. (2015) 'Hidden technical debt in machine learning systems', in Advances in Neural Information Processing Systems 28. [Online] pp. 2503–2511. Available from: https://proceedings.neurips.cc/paper/2015/hash/86df7dcfd896fcaf2674f757a2463eba-Abstract.html [Accessed 3 April 2026].

Soldner, F., Kleinberg, B. and Johnson, (2022) Confounds and Overestimations in Fake Review Detection: Experimentally Controlling For Product-ownership and Data-origin. Plos One [online]. 17 (12) [Accessed 03 April 2026].

Subramanya, R., Bhatt, D. and Bhansali, P. (2023) Managed geo-distributed feature store: architecture and system design. [Preprint] Available from: https://arxiv.org/abs/2305.20077 [Accessed 28 April 2026].

Talaat, A.S. and , (2023) Sentiment Analysis Classification System Using Hybrid Bert Models. Journal of Big Data [online]. 10, p. 110. [Accessed 23 April 2026].

Uchendu, A., Le, T. and Lee, D. (2020) 'Authorship attribution for neural text generation', in Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP). [Online] pp. 8384–8395. Available from: https://aclanthology.org/2020.emnlp-main.673 [Accessed 28 April 2026].

Wang, S. and Manning, C. (2012) 'Baselines and bigrams: simple, good sentiment and topic classification', in Proceedings of the 50th Annual Meeting of the Association for Computational Linguistics. [Online] 2, pp. 90–94. Available from: https://aclanthology.org/P12-2018 [Accessed 25 February 2026].

Yao, W., Dai, Z., Huang, R. and Caverlee, J. (2017) 'Online deception detection refueled by real world data collection', in Proceedings of the International Conference Recent Advances in Natural Language Processing, RANLP 2017. [Online] pp. 793–802. Available from: doi.org [Accessed 18 February 2026]

Yuan, H., Liu, M., Kang, L., Miao, C., Wu, Y., Li, X. and Krauthammer, M. (2023) 'An empirical study of the effect of background data size on the stability of SHapley Additive exPlanations (SHAP) for deep learning models', in The Eleventh International Conference on Learning Representations. [Online] pp. 1–2. Available from: https://openreview.net/forum?id=L38bbHmRKx [Accessed 25 April 2026].

Zellers, R., Holtzman, A., Rashkin, H., Bisk, Y., Farhadi, A., Roesner, F. and Choi, Y. (2019) 'Defending against neural fake news', in Advances in Neural Information Processing Systems 32. [Online] pp. 9051–9062. Available from: https://proceedings.neurips.cc/paper/2019/hash/3e9f0fc9b2f89e043bc6233994dfcf76-Abstract.html [Accessed 10 February 2026].

Zhang, X., Zhao, J. and LeCun, Y. (2015) 'Character-level convolutional networks for text classification', in Advances in Neural Information Processing Systems 28. [Online] pp. 649–657. Available from: https://proceedings.neurips.cc/paper/2015/hash/250cf8b51c773f3f8dc8b4be867a9a02-Abstract.html [Accessed 10 February 2026].







































