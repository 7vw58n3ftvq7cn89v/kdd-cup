
judge_info_usefulness_prompt = """

"""

summarize_rag_info_prompt = """

"""

judge_rag_info_prompt = """
## Objective:
Evaluate if retrieved text (T) provides additional useful information beyond existing summary (S) to answer question (Q).

## Instructions:
Compare T with S to identify new relevant information that helps answer Q.
Only consider T relevant if it contains unique facts not mentioned in S, and the facts are useful for answering Q.

## Output Format (Strict):
Analysis:  
[Brief analysis about T - max 2 sentences]  

Is_relevant: [Yes/No]  

Summary:
[New useful information from T not present in S; empty if not relevant]
""".strip()


judge_rag_info_prompt_few_shot = """
Examples:
Example 1 (Relevant):

Q: "What is the breed of the dog in the image?"T: "Labradors are known for their intelligence and often work as service dogs."Output:Analysis:  
The text identifies Labradors as a specific breed with recognized traits.  

Is_relevant: Yes  

Summary:  
The text explicitly mentions 'Labradors' as a breed, which directly answers the question about the dog's breed.  


Example 2 (Irrelevant):

Q: "What is the price of the product in the image?"T: "Coffee shops in Paris offer outdoor seating."Output:Analysis:  
The text discusses Parisian coffee shops, unrelated to product pricing.  

Is_relevant: No  


Example 3 (Partial Relevance):

Q: "What activity is happening in the image?"T: "Outdoor concerts often feature live bands and large crowds."Output:Analysis:  
The text describes a type of event involving crowds, which could be related to group activities.  

Is_relevant: Yes  

Summary:  
While not explicitly mentioning the image content, the text highlights 'outdoor concerts' as an event type that may involve similar dynamics.  


Constraints:

Only summarize content from T that directly supports answering Q.Avoid speculative language (e.g., "may be relevant") unless T provides ambiguous information.If T contains multiple relevant points, prioritize the most specific/impactful ones.

"""


check_answer_prompt = """
## Objective:
Evaluate whether the response (A) directly and clearly answers the question (Q) without ambiguity, and shows no doubt or uncertainty in the answer.

## Output Format (Strict):
Evaluation:  
[Your analysis here – max 2 sentences]  

Conclusion: [Yes/No]  
"""


# not use
summarize_prompt = """
## Objective:
Given the image (I) and the question (Q), please analyze:
1.Carefully examine the image to extract all relevant visual details, textual elements, or contextual clues that could potentially aid in answering the question.
2.Analyze whether the extracted information is sufficient to provide a complete and accurate answer to the question.
3.If the information is insufficient, identify and summarize the specific missing details required to address the question.

## Output Format (Strict):
Extract Information: [extract information from image and analyze whether it's enough for the question].
Missing Information: [Summarize specific details required for answering the question,simply answer **no** if information is enough].
"""

generate_subquestion_prompt = """
## Role:
You are an analytical agent responsible for identifying information gaps and generating targeted follow-up questions.

## Objective:
Based on the question Q and existing information S, identify ONE key aspect that still needs to be addressed.
Generate ONE specific sub-question to fill this information gap.

## Instructions:
1. Analyze the gap between Q and S carefully
2. Focus on the most critical missing information
3. Generate a clear, specific sub-question
4. Avoid redundant or already answered aspects

## Output Format (Strict):
Analysis: [Brief analysis of what critical information is missing - 1 sentence]
Sub-question: [Single most important follow-up question - 1 sentence]
"""

judge_ready_prompt = """
## Objective:
Given a question Q and the current information S, Think step by step about whether it is sufficient for you to answer Q base on the information S. 
Provide a concise analysis and a strict yes/no response.

## Output Format (Strict):
Analysis: [Concise reasoning on whether S contains enough information for you to answer Q]  
Ready to answer: [yes/no] 
"""