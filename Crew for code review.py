#!/usr/bin/env python3
# Crew AI Code Review for C - Workshop Demo
# This demo shows how to build a multi-agent system for C code review using Crew AI
pip install crewai>=0.28.0 langchain>=0.0.267 langchain-openai>=0.0.2 python-dotenv>=1.0.0 openai>=1.3.0

import os
import sys
from pathlib import Path
from crewai import Agent, Task, Crew, Process
from langchain_openai import ChatOpenAI

# Set your OpenAI API key directly here
# Replace "your-api-key-here" with your actual OpenAI API key
os.environ["OPENAI_API_KEY"] = "OpenAI_API_KEY"

# Define the model to use (gpt-4o-mini as requested)
MODEL = "gpt-4o-mini"

# Sample C code to review (loaded from file or defined inline)
SAMPLE_C_CODE_PATH = Path("sample_code.c")
sample_code = ""

try:
    sample_code = SAMPLE_C_CODE_PATH.read_text()
except FileNotFoundError:
    # Fallback to a sample code if file doesn't exist
    print("Sample file not found, using built-in example")
    sample_code = """
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Function to process user input and store in buffer
void process_input(char *user_input) {
    char buffer[10];
    strcpy(buffer, user_input);  // Copy user input to buffer
    printf("Processed input: %s\\n", buffer);
}

// Function to convert string to integer
int convert_to_int(char *str) {
    return atoi(str);
}

// Function to read a file
void read_file(char *filename) {
    FILE *file = fopen(filename, "r");
    if(file) {
        char buffer[100];
        while(fgets(buffer, 100, file)) {
            printf("%s", buffer);
        }
    }
    // Note: file is not closed if successfully opened
}

// Main function
int main(int argc, char *argv[]) {
    // Check if at least one argument is provided
    if(argc < 2) {
        printf("Usage: %s <input>\\n", argv[0]);
        return 1;
    }
    
    // Process the input
    process_input(argv[1]);
    
    // Convert the input to integer and print
    int num = convert_to_int(argv[1]);
    printf("Number: %d\\n", num);
    
    // Open a hardcoded file
    read_file("data.txt");
    
    int *arr = (int*)malloc(5 * sizeof(int));
    arr[0] = 10;
    arr[1] = 20;
    // Rest of array not initialized
    
    printf("Value: %d\\n", arr[2]);  // Using uninitialized memory
    
    // Memory leak - arr is not freed

    return 0;
}
"""
    # Create the sample file for future use
    SAMPLE_C_CODE_PATH.write_text(sample_code)
    print(f"Created sample code file at {SAMPLE_C_CODE_PATH}")

# Define custom C coding guidelines
CODING_GUIDELINES = """
# C Coding Guidelines

## Memory Management
- Always free any dynamically allocated memory
- Check return values of malloc() and related functions
- Avoid memory leaks by tracking all allocations

## Buffer Safety
- Never use strcpy(), strcat() without bounds checking
- Prefer safer alternatives like strncpy(), strncat()
- Always ensure buffers are properly sized for inputs

## File Operations
- Always close files after opening them
- Check return values of file operations
- Handle error conditions appropriately

## Input Validation
- Validate all user inputs before processing
- Never trust input size - always check boundaries
- Sanitize inputs to prevent injection attacks

## Code Structure
- Avoid global variables when possible
- Use consistent indentation (4 spaces preferred)
- Document functions with purpose and parameters

## Variable Initialization
- Initialize all variables before use
- Don't rely on default initialization
- Be explicit about variable scope and lifetime
"""
def create_code_review_crew():
    """Define the specialist agents for our code review crew"""
    
    # Initialize the LLM
    llm = ChatOpenAI(model=MODEL)
    
    # Agent 1: Security Specialist
    security_specialist = Agent(
        role="Security Specialist",
        goal="Identify and fix security vulnerabilities in C code",
        backstory="An expert in security vulnerabilities with 15 years of experience in analyzing C code for security issues.",
        verbose=True,
        allow_delegation=True,
        llm=llm
    )

    # Agent 2: Memory Management Specialist
    memory_specialist = Agent(
        role="Memory Management Specialist",
        goal="Identify memory leaks, buffer overflows, and improper memory handling",
        backstory="A C language expert who specializes in memory management issues and best practices.",
        verbose=True,
        allow_delegation=True,
        llm=llm
    )

    # Agent 3: Code Style & Guidelines Checker
    style_specialist = Agent(
        role="Code Style & Guidelines Specialist",
        goal="Ensure code adheres to provided style guidelines and best practices",
        backstory="A veteran programmer who has written and reviewed millions of lines of C code across multiple organizations.",
        verbose=True,
        allow_delegation=True,
        llm=llm
    )

    # Agent 4: Code Optimizer
    optimization_specialist = Agent(
        role="Code Optimization Specialist",
        goal="Identify performance issues and suggest optimizations",
        backstory="An expert in C performance optimization with a background in systems programming and compiler design.",
        verbose=True,
        allow_delegation=True,
        llm=llm
    )

    # Agent 5: Technical Writer (for the final report)
    technical_writer = Agent(
        role="Technical Writer",
        goal="Create clear, concise, and actionable reports of code issues and improvements",
        backstory="A technical writer with experience in documenting code reviews and creating developer-friendly documentation.",
        verbose=True,
        allow_delegation=True,
        llm=llm
    )

    # Create tasks for our crew
    security_review_task = Task(
        description=f"""
            Review the following C code for security vulnerabilities:
            
            {sample_code}
            
            Identify any security issues such as buffer overflows, improper input validation, 
            or other security concerns. Be specific about line numbers and exact issues.
            Reference any violations of these guidelines:
            
            {CODING_GUIDELINES}
        """,
        agent=security_specialist,
        expected_output="A detailed report of security vulnerabilities found in the code"
    )

    memory_review_task = Task(
        description=f"""
            Review the following C code for memory management issues:
            
            {sample_code}
            
            Look for memory leaks, uninitialized memory use, improper allocation/deallocation,
            buffer overflows, and similar memory-related problems. Be specific about line numbers and exact issues.
            Reference any violations of these guidelines:
            
            {CODING_GUIDELINES}
        """,
        agent=memory_specialist,
        expected_output="A detailed report of memory management issues found in the code"
    )

    style_review_task = Task(
        description=f"""
            Review the following C code for adherence to coding guidelines:
            
            {sample_code}
            
            Check if the code follows these guidelines:
            
            {CODING_GUIDELINES}
            
            Focus on code structure, documentation, naming conventions, and overall readability.
            Be specific about line numbers and exact issues.
        """,
        agent=style_specialist,
        expected_output="A detailed report of style and guideline violations found in the code"
    )

    optimization_review_task = Task(
        description=f"""
            Review the following C code for optimization opportunities:
            
            {sample_code}
            
            Identify any performance bottlenecks, inefficient algorithms, or other areas
            where the code could be optimized. Be specific about line numbers and exact issues.
        """,
        agent=optimization_specialist,
        expected_output="A detailed report of optimization opportunities found in the code"
    )

    final_report_task = Task(
        description=f"""
            Create a comprehensive code review report based on the findings from the security,
            memory management, style, and optimization reviews.
            
            Original code:
            {sample_code}
            
            Organize the report into sections:
            1. Executive Summary
            2. Critical Issues (security, memory, etc.)
            3. Style and Best Practice Issues
            4. Optimization Opportunities
            5. Recommended Fixed Code
            
            The report should be clear, actionable, and professional.
        """,
        agent=technical_writer,
        expected_output="A comprehensive code review report with fixed code suggestions",
        dependencies=[
            security_review_task, 
            memory_review_task, 
            style_review_task, 
            optimization_review_task
        ]
    )

    # Create the crew
    crew = Crew(
        agents=[
            security_specialist, 
            memory_specialist, 
            style_specialist, 
            optimization_specialist, 
            technical_writer
        ],
        tasks=[
            security_review_task, 
            memory_review_task, 
            style_review_task, 
            optimization_review_task, 
            final_report_task
        ],
        verbose=True,  # Set verbosity as boolean
        process=Process.sequential  # Run tasks in order based on dependencies
    )

    return crew

def run_code_review_demo():
    """Main function to run the demo"""
    print("Starting C Code Review Crew AI Demo")
    print("====================================")
    
    print("\nLoading code for review...")
    print(f"Code sample length: {len(sample_code)} characters")
    
    print("\nLoading coding guidelines...")
    print(f"Guidelines length: {len(CODING_GUIDELINES)} characters")
    
    print("\nCreating agent crew...")
    crew = create_code_review_crew()
    
    print("\nRunning code review tasks...")
    result = crew.kickoff()
    
    print("\nCode Review Completed!")
    print("=====================")
    print(result)
    
    # Extract the result content as string
    # In the latest versions of crewAI, we need to access the final task's output
    result_content = str(result)
    
    # Write the results to a file for easier viewing
    with open('code_review_results.md', 'w') as f:
        f.write(result_content)
    print("\nResults saved to code_review_results.md")

if __name__ == "__main__":
    # Verify API key is set correctly
    if os.environ.get("OPENAI_API_KEY") == "your-api-key-here":
        print("ERROR: Please replace 'your-api-key-here' in the script with your actual OpenAI API key.")
        print("Look for the line: os.environ[\"OPENAI_API_KEY\"] = \"your-api-key-here\"")
        sys.exit(1)
    
    run_code_review_demo()