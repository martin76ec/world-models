#!/bin/bash
# Documentation setup script for word-models project
# Run this to recreate the docmancer index for this project

echo "Adding core tool: uv..."
docmancer add https://docs.astral.sh/uv/

echo "Adding cognitive & agent frameworks..."
docmancer add https://langchain-ai.github.io/langgraph/
docmancer add https://huggingface.co/docs/smolagents/

echo "Adding world model & simulation engines..."
docmancer add https://jax.readthedocs.io/
docmancer add https://genesis-world.readthedocs.io/

echo "Adding alternative PyTorch targets..."
docmancer add https://github.com/pytorch/cppdocs/blob/master/llms-full.txt
docmancer add https://pytorch.org/executorch/stable/llm/index.html

echo "Indexing complete. Use 'docmancer list' to verify."
