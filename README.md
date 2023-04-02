# DetectGPT: Zero-Shot Machine Generated Text Detection Using Probability Curvature

## Paper
Mitchell, E., Lee, Y., Khazatsky, A., Manning, C. D., &amp; Finn, C. (2023, January 26). Detectgpt: Zero-shot machine-generated text detection using probability curvature. arXiv.org. 

## Overview
The fluency and factual knowledge of large language models (LLMs) heightens the need for corresponding systems to detect whether a piece of
text is machine-written. For example, students may use LLMs to complete written assignments, leaving instructors unable to accurately assess student
learning. This paper proposes a new method for detecting machine-generated text using probability curvature. The authors argue that current methods for detecting machine-generated text have limitations and can be easily bypassed by adversarial attacks. Therefore, they propose a new approach (DetectGPT) that uses the probability curvature of the text to distinguish between human-written and machine-generated text. 

**Question**: How will you identify between machine-generated text and human-written text? What possible differences could you think of between them?

## Architecture

* Perturbation using generic pre-trained model
* Caculate log probability of the perturbed sample
* Compare the log probability under p of the original sample with each perturbed sample


Input: A piece of text T to be classified as machine-generated or human-written

Output: A binary label y ∈ {0, 1} indicating whether T is machine-generated (y=1) or human-written (y=0)

1. Compute the log probability function p(T) for T using a pre-trained language model (e.g., GPT-2 or GPT-3).
2. Compute the third derivative of the log probability function, κ(T), for T.
3. Compute a curvature score s(T) as follows:
   a. If κ(T) is negative, s(T) = -κ(T).
   b. If κ(T) is positive or zero, s(T) = 0.
4. Compute the average curvature score for the text, S(T), as follows:
   S(T) = 1/|T| ∑_{i=1}^{|T|} s(T_i)
   where |T| is the length of the text and T_i is the i-th character in T.
5. Classify T as machine-generated if S(T) is below a threshold τ, otherwise classify it as human-written.

![image](https://user-images.githubusercontent.com/68664277/229377426-84ab8508-ae2f-4d7c-9e42-e788dd0a9a16.png)

## Algorithm for DetectGPT
![image](https://user-images.githubusercontent.com/68664277/229378613-46f8142f-5ad3-4ed5-b799-6905b133bdf3.png)

## Hypothesis
If q produces samples on the data manifold, d(x; p $\theta$; q)
is positive with high probability for samples x ~ p $\theta$ . 
For human-written text, d (x; p $\theta$; q) tends toward zero for all x.

Minor rewrites of model-generated text tend to have lower log probability under the model than the original sample, while minor rewrites of human-written text may have higher or lower log probability than the original sample.
![image](https://user-images.githubusercontent.com/68664277/229378254-54ddc74e-0408-4104-aa4d-308a6367e289.png)

**Question**: Why does model-generated text lie in areas where the log probability  function has negative curvature?

**Perturbation Discrepancy**:

![image](https://user-images.githubusercontent.com/68664277/229379272-0d028918-eb81-4683-bea4-85deb81cf54d.png)

## Datasets

* XSum(Narayan et al.,2018): Fake news detection
* SQuAD contexts (Rajpurkar et al., 2016): Wikipedia paragraphs
* Reddit WritingPrompts dataset (Fan et al.,2018): Machine-generated creative writing submissions
* WMT16 (Bojar et al., 2016): English and German splits
* PubMedQA dataset (Jin et al., 2019): long-form answers written by human experts


## Experiment

* Hypothesis testing

The average drop in log probability (perturbation discrepancy) after rephrasing a passage is consistently higher for model-generated passages than for human-written passages.

![image](https://user-images.githubusercontent.com/68664277/229381806-cdb30c20-2d7c-432f-a4ce-5548da98b88d.png)

* Comparisons with other existing methods for zero-shot and supervised detection
![image](https://user-images.githubusercontent.com/68664277/229382498-185fbe7f-f1ae-4af6-a925-e0c8a0c7c454.png)

* Comparison with Supervised Detectors

![image](https://user-images.githubusercontent.com/68664277/229382860-51e94257-aaa1-417d-8c11-b4068c168b87.png)

Supervised machine-generated text detection models trained on large datasets of real and generated texts perform as well as or better than DetectGPT on in-distribution (top row) text. However, zero-shot methods work out-of-the-box for new domains (bottom row) such as PubMed medical texts and German news data from WMT16.

### Experiment with variants of Machine-generated text detection

* Percentage of human edits

![image](https://user-images.githubusercontent.com/68664277/229383290-a050a979-9be9-460f-83dc-d4fd2428bd84.png)

* Change of decoding strategy (using top-k)

![image](https://user-images.githubusercontent.com/68664277/229383354-52b0276d-5db7-4755-994c-dac207a9c781.png)

* Using Likelihoods from Models other than the Source Model

![image](https://user-images.githubusercontent.com/68664277/229383388-661d64bc-8fe1-4691-a6bf-94e03cd04ab0.png)

## Limitations

* Watermarking

LLMs that do not perfectly imitate human writing essentially watermark themselves implicitly. If these rephrasings are systematically lower-probability than the original passage, the model is exposing its bias toward the specific (and roughly arbitrary, by human standards) phrasing used.

* Difficulty with retriving log probability

For models behind APIs that do provide probabilities (such as GPT-3), evaluating probabilities nonetheless costs money.

* Perturbation function variance 

Some domains may see reduced performance if existing mask-filling models do not well represent the space of meaningful rephrases, reducing the quality of the
curvature estimate.

* Compute-intensive

Requires sampling and scoring the set of perturbations for each candidate passage, rather than just the candidate passage


## Future Work

* Add watermarking biases to model outputs might improve model effectiveness
* Explore the relationship between prompting and detection

*Question* : Can a clever prompt successfully prevent a model’s generations from being detected by existing methods?

## Critical Analysis

* Lack of Diversity in the dataset
* Use of a Single Probability Curvature Measure
* Lack of Analysis on the Interpretability of Probability Curvature

## Resource Links

* Paper and its implications in OpenAI platform: [Blog Post](https://openai.com/blog/detecting-gpt-generated-text/)
* [Video Presentation](https://www.youtube.com/watch?v=xofz2SD-eZI)
* [Towards Detecting Machine-Generated Text" by Gao et al. (2021)](https://arxiv.org/abs/2101.10887)
* [Identifying GPT-2 generated text using stylometric analysis" by Bayer and O'Neill (2020)](https://arxiv.org/abs/2009.06791)
* [Catching the drift: Probabilistic content models, with applications to generation and summarization" by Das et al. (2021)](https://arxiv.org/abs/2102.00863)




