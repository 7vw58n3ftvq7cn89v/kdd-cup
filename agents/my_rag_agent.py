from typing import Dict, List, Any
import os

import torch
from PIL import Image
from agents.base_agent import BaseAgent
import agents.prompts as prompts
from cragmm_search.search import UnifiedSearchPipeline
from rich.console import Console
import vllm
import json

console = Console()

# Configuration constants
AICROWD_SUBMISSION_BATCH_SIZE = 8

# GPU utilization settings 
# Change VLLM_TENSOR_PARALLEL_SIZE during local runs based on your available GPUs
# For example, if you have 2 GPUs on the server, set VLLM_TENSOR_PARALLEL_SIZE=2. 
# You may need to uncomment the following line to perform local evaluation with VLLM_TENSOR_PARALLEL_SIZE>1. 
# os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'

#### Please ensure that when you submit, VLLM_TENSOR_PARALLEL_SIZE=1. 
VLLM_TENSOR_PARALLEL_SIZE = 1 
VLLM_GPU_MEMORY_UTILIZATION = 0.85 


# These are model specific parameters to get the model to run on a single NVIDIA L40s GPU
MAX_MODEL_LEN = 8192
MAX_NUM_SEQS = 2
MAX_GENERATION_TOKENS = 75

# Number of search results to retrieve
NUM_SEARCH_RESULTS = 4


judge_enough_info_prompt = """
Thought: [need more information or ready to answer]
Action: [Search/Answer]
"""


class RetrievalAgent:
    """
    Base class for retrieval agents.
    
    根据当前的问题描述以及检索到的信息，生成用于新一轮检索的文本内容，并进行检索
    
    Attributes:
        search_pipeline (UnifiedSearchPipeline): A pipeline for searching web and image content.
    """

    def __init__(self, search_pipeline: UnifiedSearchPipeline):
        self.search_pipeline = search_pipeline


class MyRAGAgent(BaseAgent):
    """
    SimpleRAGAgent demonstrates all the basic components you will need to create your 
    RAG submission for the CRAG-MM benchmark.
    Note: This implementation is not tuned for performance, and is intended for demonstration purposes only.
    
    This agent enhances responses by retrieving relevant information through a search pipeline
    and incorporating that context when generating answers. It follows a two-step approach:
    1. First, batch-summarize all images to generate effective search terms
    2. Then, retrieve relevant information and incorporate it into the final prompts
    
    The agent leverages batched processing at every stage to maximize efficiency.
    
    Note:
        This agent requires a search_pipeline for RAG functionality. Without it,
        the agent will raise a ValueError during initialization.
    
    Attributes:
        search_pipeline (UnifiedSearchPipeline): Pipeline for searching relevant information.
        model_name (str): Name of the Hugging Face model to use.
        max_gen_len (int): Maximum generation length for responses.
        llm (vllm.LLM): The vLLM model instance for inference.
        tokenizer: The tokenizer associated with the model.
    """

    def __init__(
        self, 
        search_pipeline: UnifiedSearchPipeline, 
        model_name: str = "meta-llama/Llama-3.2-11B-Vision-Instruct", 
        max_gen_len: int = 64
    ):
        """
        Initialize the RAG agent with the necessary components.
        
        Args:
            search_pipeline (UnifiedSearchPipeline): A pipeline for searching web and image content.
                Note: The web-search will be disabled in case of Task 1 (Single-source Augmentation) - so only image-search can be used in that case.
                      Hence, this implementation of the RAG agent is not suitable for Task 1 (Single-source Augmentation).
            model_name (str): Hugging Face model name to use for vision-language processing.
            max_gen_len (int): Maximum generation length for model outputs.
            
        Raises:
            ValueError: If search_pipeline is None, as it's required for RAG functionality.
        """
        super().__init__(search_pipeline)
        
        if search_pipeline is None:
            raise ValueError("Search pipeline is required for RAG agent")
            
        self.model_name = model_name
        self.max_gen_len = max_gen_len
        self.judge_system_prompt = prompts.judge_rag_info_prompt
        self.check_answer_prompt = prompts.check_answer_prompt
        
        self.initialize_models()

        # 初始化记录文件
        with open("data/filtered_results.json", "w", encoding="utf-8") as f:
            f.write('')
        self.prompts_and_responses_path = "data/prompts_and_responses.txt"
        with open(self.prompts_and_responses_path, "w") as f:
            f.write('')
        self.image_summaries_path = "data/image_summaries.json"
        with open(self.image_summaries_path, "w") as f:
            f.write('')
        self.subquestions_path = "data/subquestions.json"
        with open(self.subquestions_path, "w") as f:
            f.write('')
        self.judge_ready_path = "data/judge_ready.json" 
        with open(self.judge_ready_path, "w") as f:
            f.write('')


        
    def initialize_models(self):
        """
        Initialize the vLLM model and tokenizer with appropriate settings.
        
        This configures the model for vision-language tasks with optimized
        GPU memory usage and restricts to one image per prompt, as 
        Llama-3.2-Vision models do not handle multiple images well in a single prompt.
        
        Note:
            The limit_mm_per_prompt setting is critical as the current Llama vision models
            struggle with multiple images in a single conversation.
            Ref: https://huggingface.co/meta-llama/Llama-3.2-11B-Vision-Instruct/discussions/43#66f98f742094ed9e5f5107d4
        """
        print(f"Initializing {self.model_name} with vLLM...")
        
        # Initialize the model with vLLM
        self.llm = vllm.LLM(
            self.model_name,
            tensor_parallel_size=VLLM_TENSOR_PARALLEL_SIZE, 
            gpu_memory_utilization=VLLM_GPU_MEMORY_UTILIZATION, 
            max_model_len=MAX_MODEL_LEN,
            max_num_seqs=MAX_NUM_SEQS,
            trust_remote_code=True,
            dtype="bfloat16",
            enforce_eager=True,
            limit_mm_per_prompt={
                "image": 1 
            } # In the CRAG-MM dataset, every conversation has at most 1 image
        )
        self.tokenizer = self.llm.get_tokenizer()
        
        print("Models loaded successfully")

    def get_batch_size(self) -> int:
        """
        Determines the batch size used by the evaluator when calling batch_generate_response.
        
        The evaluator uses this value to determine how many queries to send in each batch.
        Valid values are integers between 1 and 16.
        
        Returns:
            int: The batch size, indicating how many queries should be processed together 
                 in a single batch.
        """
        return AICROWD_SUBMISSION_BATCH_SIZE
    
    def batch_summarize_images(self, images: List[Image.Image], queries:List[str]) -> List[str]:
        """
        Generate brief summaries for a batch of images to use as search keywords.
        
        This method efficiently processes all images in a single batch call to the model,
        resulting in better performance compared to sequential processing.
        
        Args:
            images (List[Image.Image]): List of images to summarize.
            
        Returns:
            List[str]: List of brief text summaries, one per image.
        """
        # Prepare image summarization prompts in batch
        # summarize_prompt = "Please summarize the image with one sentence that describes its key elements, consider the question for this image."
        my_summarize_prompt = "Please summarize the image with one sentence that describes its key elements that are useful for answering the given question. You don't need to answer the question."
        # system_content = "You are a helpful assistant that accurately describes images. Your responses are subsequently used to perform a web search to retrieve the relevant information about the image for answering the question."
        system_content = "You are a helpful assistant that accurately describes images."
        
        inputs = []
        for (image, query) in zip(images, queries):
            prompt = f"{my_summarize_prompt}\nQuestion: {query}"
            messages = [
                {"role": "system", "content": system_content},
                {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]},
            ]
            
            # Format prompt using the tokenizer
            formatted_prompt = self.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=False
            )
            
            inputs.append({
                "prompt": formatted_prompt,
                "multi_modal_data": {
                    "image": image
                }
            })

        
        # Generate summaries in a single batch call
        outputs = self.llm.generate(
            inputs,
            sampling_params=vllm.SamplingParams(
                temperature=0.1,
                top_p=0.9,
                max_tokens=50,  # Short summary only
                skip_special_tokens=True
            )
        )
        
        # Extract and clean summaries
        summaries = [output.outputs[0].text.strip() for output in outputs]
        # console.print(f"summaries: {summaries}")
        print(f"Generated {len(summaries)} image summaries")

        
        for i, (query, summary) in enumerate(zip(queries, summaries)):
            with open("data/image_summaries.json", "a", encoding="utf-8") as f:
                data = {
                    "query": query,
                    "image_summary": summary
                }
                json.dump(data, f, ensure_ascii=False, indent=4)
                # f.write(f"[query {i}]:{queries[i]}\n## Prompt:\n{prompt}\n\n## Response:\n{response}\n\n")
                f.write("\n")
        return summaries
    
    def batch_summarize_images_v2(self, images: List[Image.Image], queries:List[str]) -> List[str]:
        """
         extract information from image and analyze whether it's enough for the question.
         return args: list of dicts
            {
                extracted_info: str,
                missing_info: str
            }
        """
        # Prepare image summarization prompts in batch
        # summarize_prompt = "Please summarize the image with one sentence that describes its key elements, consider the question for this image."
        system_content = prompts.summarize_prompt
        
        inputs = []
        for (image, query) in zip(images, queries):
            prompt = f"Question: {query}"
            messages = [
                {"role": "system", "content": system_content},
                {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]},
            ]
            
            # Format prompt using the tokenizer
            formatted_prompt = self.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=False
            )
            
            inputs.append({
                "prompt": formatted_prompt,
                "multi_modal_data": {
                    "image": image
                }
            })

        # Generate summaries in a single batch call
        outputs = self.llm.generate(
            inputs,
            sampling_params=vllm.SamplingParams(
                temperature=0.1,
                top_p=0.9,
                max_tokens=100,
                skip_special_tokens=True
            )
        )
        
        # Extract and clean summaries
        responses = [output.outputs[0].text.strip() for output in outputs]
        # save the image responses
        summaries = []
        for response, query in zip(responses, queries):
            pair = {
                "query":query,
                "response": response
            }
            with open("data/summaries.json", "a", encoding="utf-8") as f:
                json.dump(pair, f, ensure_ascii=False, indent=4)
            extracted_info = response.split("Extract Information:")[-1].split("Missing Information:")[0].strip()
            missing_info = response.split("Missing Information:")[-1].strip()
            summaries.append({'extracted_info': extracted_info, 'missing_info': missing_info})
        return summaries


    def search_images(
        self, 
        queries: List[str], 
        images: List[Image.Image]
    ) -> List[List[Dict[str, Any]]]:
        """
        Perform image search using the provided queries and image summaries.
        
        This method retrieves relevant information for each query-image pair.
        
        Args:
            queries (List[str]): List of user questions.
            images (List[Image.Image]): List of images to analyze.
            image_summaries (List[str]): List of image summaries for search.
            
        Returns:
            List[List[Dict[str, Any]]]: List of search results for each query-image pair.
        """
        # Placeholder for search results
        search_results = []
        
        # Perform search for each query-image pair
        for image in images:
            image_result = self.search_pipeline(image, k=NUM_SEARCH_RESULTS)
            search_results.append(image_result)
        
        return search_results
    
    def batch_summarize_rag_info(
        self, 
        batch_search_results: List[List[str]],
        images: list[Image.Image],
        summaries: list[str],
        queries: list[str]
    )->list[list[str]]:
        """
            批量过滤并总结检索结果
            input: Q, A, S
            return: filtered S
        """

        inputs = []
        system_prompt = prompts.judge_rag_info_prompt
        for i, (search_results, query, summary,image) in enumerate(zip(batch_search_results, queries, summaries, images)):
            for snippet in search_results:
                prompt = f"##Input \nQuestion (Q): {query}\nSummary (S): {summary}\nText (T): {snippet}\n\n"
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role":"user", "content":[
                        # {"type":"image"},
                        {"type": "text", "text": prompt},
                    ]}
                ]

                formatted_prompt = self.tokenizer.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    tokenize=False
                )

                inputs.append({
                    "prompt": formatted_prompt,
                    # "multi_modal_data": {
                    #     "image": image
                    # }
                })
        
        outputs = self.llm.generate(
            inputs,
            sampling_params=vllm.SamplingParams(
                temperature=0.1,
                top_p=0.9,
                max_tokens=200, # need to adjust 
                skip_special_tokens=True
            )
        )
        outputs = [output.outputs[0].text.strip() for output in outputs]
        batch_evaluations = [outputs[i:i+NUM_SEARCH_RESULTS] for i in range(0,len(outputs),NUM_SEARCH_RESULTS)]
        batch_filtered_context = []
        for evaluations in batch_evaluations:
            filtered_context = ""
            for i, evaluation in enumerate(evaluations):
                judgement = evaluation.split("Is_relevant:")[-1].strip() if "Is_relevant" in evaluation else ''
                if "no" in judgement.lower() or not judgement:
                    continue
                info_summary = evaluation.split("Summary:")[-1].strip()
                filtered_context += f"{info_summary}\n\n"
            batch_filtered_context.append(filtered_context)

        return batch_filtered_context

    def filter_search_results(
        self, 
        search_results: List[Dict[str, Any]],
        image: Image.Image,
        summary: str,
        query: str
    ) -> str:
        """评估检索内容，并返回整理好的相关文本内容的summary"""
        # console.print(f"[green]Filtering search results for query: {query}[/green]")
        # 从results中提取出page_snippet
        result_snippets = [result.get('page_snippet', '') for result in search_results]

        # judging texts
        system_prompt = self.judge_system_prompt
        judge_inputs = []
        for snippet in result_snippets:
            prompt = f"Question (Q): {query}\nText (T): {snippet}\nSummary (S): {summary}\n\n"
            messages = [
                {"role": "system", "content": system_prompt},
                {"role":"user", "content":[
                    {"type":"image"},
                    {"type": "text", "text": prompt},
                ]}
            ]

            formatted_prompt = self.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=False
            )

            judge_inputs.append({
                "prompt": formatted_prompt,
                "multi_modal_data": {
                    "image": image
                }
            })

        outputs = self.llm.generate(
            judge_inputs,
            sampling_params=vllm.SamplingParams(
                temperature=0.1,
                top_p=0.9,
                max_tokens=150, # need to adjust 
                skip_special_tokens=True
            )
        )
        
        filtered_context = ""
        evaluations = [output.outputs[0].text.strip() for output in outputs]
        for i, evaluation in enumerate(evaluations):
            # Extract the analysis and relevance judgment
            judgement = evaluation.split("Is_relevant:")[-1].strip() if "Is_relevant" in evaluation else ''
            if "no" in judgement.lower() or not judgement:
                continue
            info_summary = evaluation.split("Summary:")[-1].strip() if "Summary" in evaluation else ''
            filtered_context += f"[Info {i+1}] {info_summary}\n\n"

        pair = {
            "query": query,
            "result_snippets": result_snippets,
            "evaluations": evaluations
        }
        with open("data/filtered_results.json", "a", encoding="utf-8") as f:
            json.dump(pair, f, ensure_ascii=False, indent=4)
        
        # breakpoint()
        return filtered_context

    def batch_search_results(
            self, 
            search_queries
    )->list[list[str]]:
        """用检索pipline批量检索增强信息"""
        results_batch = []
        for i, search_query in enumerate(search_queries):
            results = self.search_pipeline(search_query, k=NUM_SEARCH_RESULTS)
            results = [result.get('page_snippet', '') for result in results]
            results_batch.append(results)
        return results_batch 

    def prepare_rag_enhanced_inputs(
        self, 
        queries: List[str], 
        images: List[Image.Image], 
        image_summaries: List[str],
        message_histories: List[List[Dict[str, Any]]]
    ) -> List[dict]:
        """
        Prepare RAG-enhanced inputs for the model by retrieving relevant information in batch.
        
        This method:
        1. Uses image summaries combined with queries to perform effective searches
        2. Retrieves contextual information from the search_pipeline
        3. Formats prompts incorporating this retrieved information
        
        Args:
            queries (List[str]): List of user questions.
            images (List[Image.Image]): List of images to analyze.
            image_summaries (List[str]): List of image summaries for search.
            message_histories (List[List[Dict[str, Any]]]): List of conversation histories.
            
        Returns:
            List[dict]: List of input dictionaries ready for the model.
        """
        # Batch process search queries
        search_results_batch = []
        filtered_results_batch = []
        
        # Create combined search queries for each image+query pair
        search_queries = [f"{query} {summary}" for query, summary in zip(queries, image_summaries)]
        
        # Retrieve relevant information for each query
        for i, search_query in enumerate(search_queries):
            results = self.search_pipeline(search_query, k=NUM_SEARCH_RESULTS)
            search_results_batch.append(results)
            filtered_results = self.filter_search_results(search_results=results, image=images[i], summary=image_summaries[i], query=queries[i]) 
            filtered_results_batch.append(filtered_results)
        
        # Prepare formatted inputs with RAG context for each query
        inputs = []
        for idx, (query, image, message_history, filtered_results) in enumerate(
            zip(queries, images, message_histories, filtered_results_batch)
        ):
            # Create system prompt with RAG guidelines
            SYSTEM_PROMPT = ("You are a helpful assistant that truthfully answers user questions about the provided image."
                           "Keep your response concise and to the point. If you don't know the answer or don't have enough inormation to answer the question, directly respond 'I don't know' without extra text.")
            
            # Add retrieved context if available
            rag_context = ""
            # # naive: directly use the search results
            # if search_results:
            #     rag_context = "Here is some additional information that may help you answer:\n\n"
            #     for i, result in enumerate(search_results):
            #         snippet = result.get('page_snippet', '')
            #         if snippet:
            #             rag_context += f"[Info {i+1}] {snippet}\n\n"

            if filtered_results:    
                rag_context = f"Here is some additional information that may help you answer:\n\n{filtered_results}"
                
            # Structure messages with image and RAG context
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": [{"type": "image"}]}
            ]
            
            # Add conversation history for multi-turn conversations
            if message_history:
                messages = messages + message_history
                
            # Add RAG context as a separate user message if available
            if rag_context:
                messages.append({"role": "user", "content": rag_context})
                
            # Add the current query
            messages.append({"role": "user", "content": query})
            
            # Apply chat template
            formatted_prompt = self.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=False
            )
            # console.print(f"[yellow]messages:{messages},[/yellow]\n[red]prompt:{formatted_prompt}[/red]")
            
            inputs.append({
                "prompt": formatted_prompt,
                "multi_modal_data": {
                    "image": image
                }
            })
        
        return inputs
    
    def check_answers(
        self, 
        answers:List[str],
        queries: List[str]
    )-> List[str]:
        system_content = self.check_answer_prompt
        inputs = []
        for (query, answer) in zip(queries, answers):
            prompt = f"Q: {query}\nA: {answer}"
            messages = [
                {"role": "system", "content": system_content},
                {"role": "user", "content": [{"type": "text", "text": prompt}]},
            ]
            
            # Format prompt using the tokenizer
            formatted_prompt = self.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=False
            )
            
            inputs.append({
                "prompt": formatted_prompt
            })
        
        # Generate summaries in a single batch call
        outputs = self.llm.generate(
            inputs,
            sampling_params=vllm.SamplingParams(
                temperature=0.1,
                top_p=0.9,
                max_tokens=80, 
                skip_special_tokens=True
            )
        )
        
        # Extract and clean summaries
        evaluations = [output.outputs[0].text.strip() for output in outputs]
        conclusions = [eval.split("Conclusion:")[-1].strip() for eval in evaluations]

        # checked_answers = ["I don't know" if "i don't know" in r.lower() else r for r in answers] 
        checked_answers = ["I don't know" if 'no' in conclusion.lower() else answer for conclusion,answer in zip(conclusions,answers)] 

        return checked_answers

    def batch_generate_response_old(
        self,
        queries: List[str],
        images: List[Image.Image],
        message_histories: List[List[Dict[str, Any]]],
    ) -> List[str]:
        """
        Generate RAG-enhanced responses for a batch of queries with associated images.
        
        This method implements a complete RAG pipeline with efficient batch processing:
        1. First batch-summarize all images to generate search terms
        2. Then retrieve relevant information using these terms
        3. Finally, generate responses incorporating the retrieved context
        
        Args:
            queries (List[str]): List of user questions or prompts.
            images (List[Image.Image]): List of PIL Image objects, one per query.
                The evaluator will ensure that the dataset rows which have just
                image_url are populated with the associated image.
            message_histories (List[List[Dict[str, Any]]]): List of conversation histories,
                one per query. Each history is a list of message dictionaries with
                'role' and 'content' keys in the following format:
                
                - For single-turn conversations: Empty list []
                - For multi-turn conversations: List of previous message turns in the format:
                  [
                    {"role": "user", "content": "first user message"},
                    {"role": "assistant", "content": "first assistant response"},
                    {"role": "user", "content": "follow-up question"},
                    {"role": "assistant", "content": "follow-up response"},
                    ...
                  ]
                
        Returns:
            List[str]: List of generated responses, one per input query.
        """
        print(f"Processing batch of {len(queries)} queries with RAG")
        # breakpoint()
        # related_images = self.search_images(queries, images)
        # import pdb; pdb.set_trace()
        
        # Step 1: Batch summarize all images for search terms
        image_summaries = self.batch_summarize_images(images,queries)
        # pdb.set_trace()


        # Step 2: Prepare RAG-enhanced inputs in batch
        rag_inputs = self.prepare_rag_enhanced_inputs(
            queries, images, image_summaries, message_histories
        )
        # pdb.set_trace()
        
        # Step 3: Generate responses using the batch of RAG-enhanced prompts
        print(f"Generating responses for {len(rag_inputs)} queries")
        outputs = self.llm.generate(
            rag_inputs,
            sampling_params=vllm.SamplingParams(
                temperature=0.1,
                top_p=0.9,
                max_tokens=MAX_GENERATION_TOKENS,
                skip_special_tokens=True
            )
        )
        
        # Extract and return the generated responses
        responses = [output.outputs[0].text for output in outputs]
        # responses = self.check_answers(responses)
        print(f"Successfully generated {len(responses)} responses")
        # pdb.set_trace()
        return responses

    def batch_judge_ready(
            self,
            images: list[Image.Image],
            queries: list[str],
            rag_summaries: list[str],
            indices: list[int]
    )->list[int]:
        if len(indices) != len(queries):
            raise ValueError(f"indices count:{len(indices)} !=  queries count :{len(queries)}")
        # ask llm to judge
        system_content = prompts.judge_ready_prompt 
        inputs = []
        for (query, summary, image) in zip(queries, rag_summaries, images):
            prompt = f"Question: {query}\nCollected information (S): {summary}\n"
            messages = [
                {"role": "system", "content": system_content},
                {"role": "user", "content": [{"type": "text", "text": prompt}]},
                # {"type": "image"}
            ]
            
            formatted_prompt = self.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=False
            )
            
            inputs.append({
                "prompt": formatted_prompt
            })
        
        # Generate summaries in a single batch call
        outputs = self.llm.generate(
            inputs,
            sampling_params=vllm.SamplingParams(
                temperature=0.1,
                top_p=0.9,
                max_tokens=200, 
                skip_special_tokens=True
            )
        )
        
        # Extract and clean summaries
        outputs = [output.outputs[0].text.strip() for output in outputs]
        judgements = [output.split("Ready to answer:")[-1].strip() for output in outputs]
        
        # update state
        need_more_info = []
        for idx, judgement in zip(indices, judgements):
            if "no" in judgement.lower():
                need_more_info.append(idx)

        # save results
        for i, (output, judgement) in enumerate(zip(outputs,judgements)):
            data = {
                'query': queries[i],
                'rag_summaries': rag_summaries[i],
                'judgement': judgement,
                'output': output
            }
            with open(self.judge_ready_path, "a") as f:
                json.dump(data, f, ensure_ascii=False, indent=4)

        return need_more_info
    
    def batch_get_augment_info(
            self,
            queries,
            rag_summaries,
            images
    )->list[str]:
        # 检索新一轮信息
        batch_search_results = self.batch_retrieve(queries, rag_summaries)
        # 总结新一轮增强信息
        new_summaries = self.batch_summarize_rag_info(
            batch_search_results=batch_search_results,
            images=images,
            summaries=rag_summaries,
            queries=queries
        )

        return new_summaries
    
    def batch_generate_subquestions(
            self,
            queries:List[str],
            rag_summaries:List[str]
    )->list[str]:
        # system_content = "根据现有问题Q，收集的信息S，判断还需要补充什么信息回答问题，并基于需要的信息，用一句话给出可用于高维向量检索的文本内容"
        system_content = prompts.generate_subquestion_prompt
        inputs = []
        for (query, summary) in zip(queries, rag_summaries):
            prompt = f"Question (Q): {query}\nexisting information (S): {summary}\n"
            messages = [
                {"role": "system", "content": system_content},
                {"role": "user", "content": [{"type": "text", "text": prompt}]},
            ]
            
            formatted_prompt = self.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=False
            )
            
            inputs.append({
                "prompt": formatted_prompt
            })

        outputs = self.llm.generate(
            inputs,
            sampling_params=vllm.SamplingParams(
                temperature=0.1,
                top_p=0.9,
                max_tokens=100, 
                skip_special_tokens=True
            )
        )
        
        outputs = [output.outputs[0].text.strip() for output in outputs]
        # 提取提示词中的text
        subquestions = []
        for i, (output, query) in enumerate(zip(outputs, queries)):
            subquestion = query
            if "Sub-question:" in output:
                subquestion = output.split("Sub-question:")[-1]
            else:
                console.print(f"[red]warning: no subquestion for query:{query}[/red]")
                subquestion = query # 用原始问题代替生成的子问题
            subquestions.append(subquestion)
            # save subquestion
            data = {
                'query':query,
                'rag_summaries': rag_summaries[i],
                'subquestion': subquestion,
                'output': output
            }
            with open(self.subquestions_path, "a") as f:
                json.dump(data, f, ensure_ascii=False, indent=4)

        return subquestions

    def batch_retrieve(
            self,
            queries,
            rag_summaries
    ):
        # 向llm提问，得到用于检索的文本
        retrieve_texts = self.batch_generate_subquestions(queries, rag_summaries)

        # 检索新一轮信息
        batch_results = self.batch_search_results(search_queries=retrieve_texts)
        
        return batch_results

    def batch_generate_response(
        self,
        queries: List[str],
        images: List[Image.Image],
        message_histories: List[List[Dict[str, Any]]],
    ) -> List[str]:
        """
        Generate RAG-enhanced responses for a batch of queries with associated images.
        
        This method implements a complete RAG pipeline with efficient batch processing:
        1. First batch-summarize all images to generate search terms
        2. Then retrieve relevant information using these terms
        3. Finally, generate responses incorporating the retrieved context
        
        Args:
            queries (List[str]): List of user questions or prompts.
            images (List[Image.Image]): List of PIL Image objects, one per query.
                The evaluator will ensure that the dataset rows which have just
                image_url are populated with the associated image.
            message_histories (List[List[Dict[str, Any]]]): List of conversation histories,
                one per query. Each history is a list of message dictionaries with
                'role' and 'content' keys in the following format:
                
                - For single-turn conversations: Empty list []
                - For multi-turn conversations: List of previous message turns in the format:
                  [
                    {"role": "user", "content": "first user message"},
                    {"role": "assistant", "content": "first assistant response"},
                    {"role": "user", "content": "follow-up question"},
                    {"role": "assistant", "content": "follow-up response"},
                    ...
                  ]
                
        Returns:
            List[str]: List of generated responses, one per input query.
        """
        print(f"Processing batch of {len(queries)} queries with RAG")

        rag_summaries = self.batch_summarize_images(images,queries)

        need_more_info = self.batch_judge_ready(images, queries, rag_summaries,indices=[i for i in range(len(queries))])

        max_turn = 3
        while need_more_info and max_turn:
            console.print(f"[red]round{4-max_turn}, with {len(need_more_info)} unready queries.[/red]")
            retrieve_queires = [queries[i] for i in need_more_info]
            retrieve_images = [images[i] for i in need_more_info]
            retrieve_summaries = [rag_summaries[i] for i in need_more_info]
            new_infos = self.batch_get_augment_info(
                queries=retrieve_queires,
                rag_summaries=retrieve_summaries,
                images=retrieve_images
            )
            # add new info to rag summaries
            for idx, new_info in zip(need_more_info,new_infos):
                if new_infos:
                    rag_summaries[idx] += f"\n[info {4-max_turn}]:{new_info}\n"
            
            batch_queries = [queries[i] for i in need_more_info]
            batch_images = [images[i] for i in need_more_info]
            batch_summaries = [rag_summaries[i] for i in need_more_info]
            need_more_info = self.batch_judge_ready(batch_images, batch_queries, batch_summaries, need_more_info)
            max_turn -= 1
                
        # Step 2: Prepare inputs in batch
        inputs = self.prepare_simple_inputs(
            queries, images, rag_summaries, message_histories
        )
        
        # Step 3: Generate responses using the batch of RAG-enhanced prompts
        print(f"Generating responses for {len(inputs)} queries")
        outputs = self.llm.generate(
            inputs,
            sampling_params=vllm.SamplingParams(
                temperature=0.1,
                top_p=0.9,
                max_tokens=MAX_GENERATION_TOKENS,
                skip_special_tokens=True
            )
        )
        
        # Extract and return the generated responses 
        responses = [output.outputs[0].text for output in outputs]
        responses = [response.lower().replace("\"", '') for response in responses]

        # Save prompts and responses to file
        for i, (prompt, response) in enumerate(zip([inp["prompt"] for inp in inputs], responses)):
            with open("data/prompts_and_responses.txt", "a", encoding="utf-8") as f:
                # json.dump(data, f, ensure_ascii=False, indent=4)
                f.write(f"[query {i}]:{queries[i]}\n## Prompt:\n{prompt}\n\n## Response:\n{response}\n\n")
                f.write("\n")
        
        for idx in need_more_info:
            responses[idx] = "i don't know"
        # console.print(f"not confident answer count:{len(need_more_info)}")

        print(f"Successfully generated {len(responses)} responses")
        return responses

    def prepare_simple_inputs(
        self, 
        queries: List[str], 
        images: List[Image.Image], 
        rag_summaries: List[str],
        message_histories: List[List[Dict[str, Any]]]
    ) -> List[dict]:
        """
        Prepare inputs for the model in batch.
        """
        # Prepare formatted inputs with RAG context for each query
        inputs = []
        for idx, (query, image, message_history, summary) in enumerate(
            zip(queries, images, message_histories, rag_summaries)
        ):
            # Create system prompt with RAG guidelines
            SYSTEM_PROMPT = ("You are a helpful assistant that truthfully answers user questions about the provided image."
                           "Keep your response concise and to the point. If you don't know the answer or don't have enough inormation to answer the question, directly respond 'I don't know' without extra text.")
            
            rag_context = ""
            if summary:    
                rag_context = f"Here is some additional information that may help you answer:\n\n[info 0]{summary}"
                
            # Structure messages with image and RAG context
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": [{"type": "image"}]}
            ]
            
            # Add conversation history for multi-turn conversations
            if message_history:
                messages = messages + message_history
                
            # Add RAG context as a separate user message if available
            if rag_context:
                messages.append({"role": "user", "content": rag_context})
                
            # Add the current query
            messages.append({"role": "user", "content": query})
            
            # Apply chat template
            formatted_prompt = self.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=False
            )
            
            inputs.append({
                "prompt": formatted_prompt,
                "multi_modal_data": {
                    "image": image
                }
            })
        
        return inputs