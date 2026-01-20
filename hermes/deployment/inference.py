"""SageMaker inference client for deployed endpoints."""

import json
from typing import Dict, Any, Optional, List, Iterator
from loguru import logger

try:
    import boto3
    from botocore.exceptions import ClientError
    BOTO3_AVAILABLE = True
except ImportError:
    BOTO3_AVAILABLE = False
    logger.warning("boto3 not installed. Install with: pip install boto3")

from hermes.deployment.config import InferenceConfig


class SageMakerInferenceClient:
    """Client for invoking SageMaker endpoints."""
    
    def __init__(
        self,
        endpoint_name: str,
        region: str = "us-east-1",
        aws_access_key: Optional[str] = None,
        aws_secret_key: Optional[str] = None,
        inference_component_name: Optional[str] = None,
        config: Optional[InferenceConfig] = None,
    ) -> None:
        """
        Initialize SageMaker inference client.
        
        Args:
            endpoint_name: SageMaker endpoint name
            region: AWS region
            aws_access_key: AWS access key (optional)
            aws_secret_key: AWS secret key (optional)
            inference_component_name: Inference component name (optional)
            config: Inference configuration
        """
        if not BOTO3_AVAILABLE:
            raise ImportError("boto3 required. Install with: pip install boto3")
        
        self.endpoint_name = endpoint_name
        self.inference_component_name = inference_component_name
        self.config = config or InferenceConfig()
        
        # Initialize SageMaker runtime client
        client_kwargs = {"region_name": region}
        if aws_access_key and aws_secret_key:
            client_kwargs["aws_access_key_id"] = aws_access_key
            client_kwargs["aws_secret_access_key"] = aws_secret_key
        
        self.client = boto3.client("sagemaker-runtime", **client_kwargs)
        
        logger.info(f"Initialized inference client for endpoint: {endpoint_name}")
    
    def _build_payload(
        self,
        inputs: str | List[str],
        parameters: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Build inference payload.
        
        Args:
            inputs: Input text or list of inputs
            parameters: Generation parameters (overrides config)
            
        Returns:
            Payload dictionary
        """
        # Start with config parameters
        payload_params = self.config.to_payload_params()
        
        # Override with custom parameters
        if parameters:
            payload_params.update(parameters)
        
        # Build payload
        payload = {
            "inputs": inputs,
            "parameters": payload_params,
        }
        
        return payload
    
    def invoke(
        self,
        inputs: str | List[str],
        parameters: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any] | List[Dict[str, Any]]:
        """
        Invoke endpoint for inference.
        
        Args:
            inputs: Input text or list of inputs
            parameters: Generation parameters
            
        Returns:
            Generated output(s)
        """
        payload = self._build_payload(inputs, parameters)
        
        logger.info(f"Invoking endpoint: {self.endpoint_name}")
        logger.debug(f"Payload: {payload}")
        
        try:
            # Prepare invoke arguments
            invoke_args = {
                "EndpointName": self.endpoint_name,
                "ContentType": "application/json",
                "Body": json.dumps(payload),
            }
            
            # Add inference component if specified
            if self.inference_component_name:
                invoke_args["InferenceComponentName"] = self.inference_component_name
            
            # Invoke endpoint
            response = self.client.invoke_endpoint(**invoke_args)
            
            # Parse response
            response_body = response["Body"].read().decode("utf-8")
            result = json.loads(response_body)
            
            logger.info("Inference successful")
            logger.debug(f"Result: {result}")
            
            return result
            
        except ClientError as e:
            logger.error(f"Inference failed: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            raise
    
    def generate(
        self,
        prompt: str,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        **kwargs,
    ) -> str:
        """
        Generate text from prompt.
        
        Args:
            prompt: Input prompt
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            **kwargs: Additional generation parameters
            
        Returns:
            Generated text
        """
        # Build parameters
        parameters = {}
        if max_new_tokens is not None:
            parameters["max_new_tokens"] = max_new_tokens
        if temperature is not None:
            parameters["temperature"] = temperature
        if top_p is not None:
            parameters["top_p"] = top_p
        parameters.update(kwargs)
        
        # Invoke endpoint
        result = self.invoke(prompt, parameters)
        
        # Extract generated text
        if isinstance(result, list) and len(result) > 0:
            generated_text = result[0].get("generated_text", "")
        elif isinstance(result, dict):
            generated_text = result.get("generated_text", "")
        else:
            generated_text = str(result)
        
        return generated_text
    
    def batch_generate(
        self,
        prompts: List[str],
        **kwargs,
    ) -> List[str]:
        """
        Generate text for multiple prompts.
        
        Args:
            prompts: List of input prompts
            **kwargs: Generation parameters
            
        Returns:
            List of generated texts
        """
        logger.info(f"Batch generating for {len(prompts)} prompts")
        
        # Invoke with batch
        results = self.invoke(prompts, kwargs)
        
        # Extract generated texts
        if isinstance(results, list):
            generated_texts = [
                r.get("generated_text", "") for r in results
            ]
        else:
            generated_texts = [str(results)]
        
        return generated_texts
    
    def stream_generate(
        self,
        prompt: str,
        **kwargs,
    ) -> Iterator[str]:
        """
        Generate text with streaming (if endpoint supports it).
        
        Note: Requires endpoint with streaming support.
        
        Args:
            prompt: Input prompt
            **kwargs: Generation parameters
            
        Yields:
            Text chunks
        """
        payload = self._build_payload(prompt, kwargs)
        
        logger.info(f"Streaming from endpoint: {self.endpoint_name}")
        
        try:
            # Prepare invoke arguments
            invoke_args = {
                "EndpointName": self.endpoint_name,
                "ContentType": "application/json",
                "Body": json.dumps(payload),
            }
            
            if self.inference_component_name:
                invoke_args["InferenceComponentName"] = self.inference_component_name
            
            # Invoke with streaming
            response = self.client.invoke_endpoint_with_response_stream(**invoke_args)
            
            # Stream response
            event_stream = response["Body"]
            for event in event_stream:
                if "PayloadPart" in event:
                    chunk = event["PayloadPart"]["Bytes"].decode("utf-8")
                    yield chunk
                    
        except ClientError as e:
            logger.error(f"Streaming inference failed: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            raise
    
    def chat(
        self,
        messages: List[Dict[str, str]],
        **kwargs,
    ) -> str:
        """
        Chat completion (if model supports chat format).
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            **kwargs: Generation parameters
            
        Returns:
            Assistant response
        """
        # Format messages into prompt
        # This is a simple implementation; adjust based on model's chat template
        prompt_parts = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            
            if role == "system":
                prompt_parts.append(f"System: {content}")
            elif role == "user":
                prompt_parts.append(f"User: {content}")
            elif role == "assistant":
                prompt_parts.append(f"Assistant: {content}")
        
        prompt_parts.append("Assistant:")
        prompt = "\n".join(prompt_parts)
        
        # Generate response
        response = self.generate(prompt, **kwargs)
        
        # Clean up response (remove any prompt repetition)
        if "Assistant:" in response:
            response = response.split("Assistant:")[-1].strip()
        
        return response
    
    def health_check(self) -> bool:
        """
        Check if endpoint is healthy.
        
        Returns:
            True if endpoint is healthy
        """
        try:
            # Try simple inference
            self.generate("Hello", max_new_tokens=10)
            logger.info(f"Endpoint {self.endpoint_name} is healthy")
            return True
        except Exception as e:
            logger.error(f"Endpoint {self.endpoint_name} health check failed: {e}")
            return False
    
    def get_endpoint_metrics(self) -> Dict[str, Any]:
        """
        Get endpoint metrics (requires CloudWatch).
        
        Returns:
            Endpoint metrics
        """
        try:
            cloudwatch = boto3.client("cloudwatch", region_name=self.client.meta.region_name)
            
            # Get common metrics
            metrics = {}
            metric_names = [
                "ModelLatency",
                "Invocations",
                "InvocationsPerInstance",
            ]
            
            for metric_name in metric_names:
                response = cloudwatch.get_metric_statistics(
                    Namespace="AWS/SageMaker",
                    MetricName=metric_name,
                    Dimensions=[
                        {
                            "Name": "EndpointName",
                            "Value": self.endpoint_name,
                        },
                    ],
                    StartTime="-PT1H",  # Last hour
                    EndTime="now",
                    Period=3600,  # 1 hour
                    Statistics=["Average", "Maximum"],
                )
                
                if response["Datapoints"]:
                    metrics[metric_name] = response["Datapoints"][0]
            
            return metrics
            
        except Exception as e:
            logger.error(f"Failed to get endpoint metrics: {e}")
            return {}


class SageMakerRAGClient(SageMakerInferenceClient):
    """Extended client for RAG-specific inference."""
    
    def query_with_context(
        self,
        query: str,
        context: str | List[str],
        system_prompt: Optional[str] = None,
        **kwargs,
    ) -> str:
        """
        Query with retrieved context (RAG pattern).
        
        Args:
            query: User query
            context: Retrieved context (string or list of strings)
            system_prompt: System prompt
            **kwargs: Generation parameters
            
        Returns:
            Generated answer
        """
        # Format context
        if isinstance(context, list):
            context_str = "\n\n".join(context)
        else:
            context_str = context
        
        # Build RAG prompt
        if system_prompt:
            prompt = f"{system_prompt}\n\n"
        else:
            prompt = "Answer the question based on the following context.\n\n"
        
        prompt += f"Context:\n{context_str}\n\n"
        prompt += f"Question: {query}\n\n"
        prompt += "Answer:"
        
        # Generate answer
        answer = self.generate(prompt, **kwargs)
        
        return answer
