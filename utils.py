from python_parsers import get_all_calls, get_all_imports, parse_commented_function, same_ast_with_reason, remove_docstring, replace_func
from get_code_docs import CodeData, get_reference_docs_custom_functions, get_shortened_docs
from prompts import SYSTEM_PROMPT, DOC_GENERATION_PROMPT
from constants import TOK_COUNT
from llm_inference import get_llm_output

import argparse
from argparse import RawTextHelpFormatter
import pandas as pd
import logging
import ast
import os
import math
import re


def get_args():
    """
    Parses and returns command line arguments for configuring the application.

    Input:
        None

    Returns:
        argparse.Namespace: Namespace object containing parsed arguments.

    Raises:
        SystemExit: If the command line arguments are invalid.
    """
    # Create an argument parser object
    # argparse.ArgumentParser: Creates a parser object to handle command-line arguments
    parser = argparse.ArgumentParser(formatter_class=RawTextHelpFormatter)
        
    # Add the path argument
    parser.add_argument(
        "path",
        help="Path to the file/folder of project",
    )
    
    # Add the verbose argument
    parser.add_argument(
        "-v", "--verbose",
        action='store_true',
        help="Give out verbose logs",
    )

    # Add the OpenAI key argument
    parser.add_argument(
        "--openai_key",
        help="Your Open AI key",
    )

    # Add the Gemini API key argument
    parser.add_argument(
        "--gemini_api_key",
        help="Your Gemini API key",
    )

    # Add the OpenAI key environment variable argument
    parser.add_argument(
        "--openai_key_env",
        help="Environment variable where Open AI key is stored",
    )

    # Add the OpenAI model argument
    parser.add_argument(
        "--openai_model",
        choices=['gpt-3.5-turbo', 'gpt-4-turbo', 'gpt-4o'],
        default='gpt-3.5-turbo',
        help="Which openAI model to use. Supported models are ['gpt-3.5-turbo', 'gpt-4-turbo', 'gpt-4o']\
            \ngpt-3.5-turbo is used by default"
    )

    # Add the Gemini model argument
    parser.add_argument(
        "--gemini_model",
        choices=['gemini-1.5-flash','gemini-1.5-flash-8b','gemini-1.5-pro','gemini-1.0-pro'],
        default='gemini-1.5-flash',
        help="Which gemini model to use. Supported models are ['gemini-1.5-flash','gemini-1.5-flash-8b','gemini-1.5-pro','gemini-1.0-pro']\
                \ngemini-1.5-flash is used by default"
    )

    # Add the port argument
    parser.add_argument(
        "-p", "--port",
        type=int,
        help="Port where Local LLM server is hosted"
    )
        
    # Add the reference documentation strategy argument
    parser.add_argument(
        "--ref_doc",
        choices=['truncate', 'summarize', 'full'],
        default='truncate',
        help="Strategy to process reference documentation. Supported choices are:\
            \ntruncate    - Truncate documentation to the first paragraph\
            \nsummarize   - Generate a single summary of the documentation using the given LLM\
            \nfull        - Use the complete documentation (Can lead to very long context length)\
            \n\"truncate\" is used as the default strategy"
    )
    
    # Add the maximum retries argument
    parser.add_argument(
        "--max_retries",
        type=int,
        default=3,
        help="Number of attempts that the LLM gets to generate the documentation for each function/method/class"
    )
    
    # Add the temperature argument
    parser.add_argument(
        "--temperature",
        type=int,
        default=0.8,
        help="Temperature parameter used to sample output from the LLM"
    )
    
    # Add the maximum tokens argument
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=2048,
        help="Maximum number of tokens that the LLM is allowed to generate"
    )

    # Parse the arguments
    args = parser.parse_args()
    # Verify the arguments parsed
    verify_args(args)
    
    return args


def verify_args(args):
    """
    Verifies the necessary command line arguments for running the LLM.

    Input:
    - args: An object containing command line arguments. It should have attributes:
        - port: Port number for local LLM (optional if using OpenAI).
        - openai_key: API key for OpenAI LLM (optional if using local LLM).
        - openai_key_env: Environment variable name for OpenAI API key (optional if using local LLM).
        - openai_model: Specific model to be used from OpenAI (optional).
        - gemini_api_key: API key for Gemini (optional).
        - gemini_model: Specific model to be used from Gemini (optional).

    Returns:
    - None

    Raises:
    - argparse.ArgumentError: Raised if neither 'port' nor 'openai_key'/'openai_key_env' is provided.
                             Raised if 'openai_model' is specified without 'openai_key' or 'openai_key_env'.
                             Raised if 'gemini_model' is specified without 'gemini_api_key'.
    """
    # Create an argument parser object
    # argparse.ArgumentParser: Object for parsing command line strings into Python objects.
    parser = argparse.ArgumentParser()

    if not (args.port or args.gemini_api_key or args.openai_key or args.openai_key_env):
        raise parser.error(
            'Must specify either --port for local LLM, or --gemini_api_key for Gemini, or --openai_key/--openai_key_env for OpenAI')

    # Check if an OpenAI model is specified without providing necessary keys
    if not args.port and not args.gemini_api_key and (args.openai_model and not (args.openai_key or args.openai_key_env)):
        raise parser.error('One of --openai_key or --openai_key_env must be specified')

    # Check if gemini model is specified without providing necessary keys
    if not args.port and not args.openai_key and (args.gemini_model and not args.gemini_api_key):
        raise parser.error('For Gemini model gemini API key must be specified')

import os
import pandas as pd


def generate_report(code_deps, input_path):
    """Generates a CSV report of code changes and documentation.

    Args:
        code_deps (dict): A dictionary containing code dependencies and their attributes.
        input_path (str): Path to the input file used for report naming.

    Returns:
        None: The function saves a CSV file to the current directory.

    Raises:
        None: No exceptions are explicitly raised by this function.
    """
    data = []  # Initialize an empty list to store the formatted data

    # Iterate through the code dependencies dictionary and extract relevant information
    # for custom code entries.
    for k, v in code_deps.items():  
        # Check if the code is custom
        if v[CodeData.CUSTOM]:  
            # Append the relevant data (path, function name, documentation, code changes) to the list
            data.append({
                'path': v['path'],
                'function': k,  # Function name
                'documentation': v[CodeData.DOC],  # Full documentation string
                'shortened documentation': v[CodeData.DOC_SHORT],  # Shortened documentation string
                'code_before': v[CodeData.CODE],  # Original code
                'code_after': v[CodeData.CODE_NEW]  # Modified code
            })

    # Extract filename from the input path using os.path.basename
    filename = os.path.basename(input_path)

    # Create the report filename by adding a prefix
    report_path = f'doc_report_{filename}'

    # Convert the list of dictionaries to a pandas DataFrame.
    # Save the DataFrame to a CSV file specified by report_path, excluding the index.
    pd.DataFrame(data).to_csv(report_path, index=False)


is_hidden_dir = lambda path: any([dir.startswith('.') for dir in path.split('/')])

def get_code_dependancies_and_imports(path):
    """
    Extracts code dependencies and import statements from Python files in a given directory or a single file.

    Input:
        path (str): The path to a directory containing Python files or a single Python file.

    Returns:
        tuple: A tuple containing:
               - code_dependancies (CodeData): An object holding the code dependencies.
               - import_stmts (list): A list of unique import statements found in the Python files.

    Raises:
        Exception: If the path is neither a directory nor a Python file.
    """
    import_stmts = []  # List to hold import statements
    code_dependancies = CodeData()  # Object to hold code dependencies
    
    if os.path.isdir(path):
        # If the path is a directory, walk through all subdirectories and files
        for root, _, files in os.walk(path):  # Iterate through the directory tree
            for file in files:  # Iterate through files in the current directory
                fpath = os.path.join(root, file)  # Construct the full file path
                if not is_hidden_dir(fpath.replace(path,"")) and os.path.splitext(file)[-1] == '.py':
                    # Process only visible Python files
                    logging.info(f'Extracting dependancies from {fpath}')  # Log the file being processed
                    
                    with open(fpath) as f: # Open the file   
                        code_str = f.read()  # Read the code from the file
                        
                    import_stmts.extend(get_all_imports(code_str)[2])  # Extract and add import statements
                    get_all_calls(fpath, code_str, code_dependancies)  # Extract and add function calls and dependencies
                    
    elif os.path.splitext(path)[-1] == '.py':
        # If the path is a single Python file
        logging.info(f'Extracting dependancies from {path}')  # Log the file being processed
        
        with open(path) as f: # Open the file    
            code_str = f.read()  # Read the code from the file
        
        import_stmts.extend(get_all_imports(code_str)[2])  # Extract and add import statements
        get_all_calls(path, code_str, code_dependancies)  # Extract and add function calls and dependencies
        
    else:
        # Raise an exception if the provided path is invalid
        raise Exception(f'Could not parse path: `{path}`') # Raise exception for invalid path
    
    import_stmts = list(set(import_stmts))  # Remove duplicate import statements
    
    return code_dependancies, import_stmts


def generate_documentation_for_custom_calls(code_dependancies, llm_mode, args):
    """
    Generate documentation for custom functions/methods/classes.

    Input:
        code_dependancies (dict): A dictionary containing function names as keys and their metadata as values.
        llm_mode (str): The mode of the language model to be used.
        args (Namespace): A namespace object containing arguments like max_retries, etc.

    Returns:
        None

    Raises:
        TypeError: If the given node does not have docstrings, a TypeError is raised.
    """
    # Fetch the list of custom functions from the code dependencies
    custom_funcs = [func_name for func_name, func_info in code_dependancies.items() if func_info[CodeData.CUSTOM]]

    # Count of custom functions and calculate number of digits for formatting progress display
    num_custom_funcs = len(custom_funcs)  
    num_digits = math.ceil(math.log(num_custom_funcs, 10))  
    logging.info(f'Generating docs for {len(custom_funcs)} custom functions/methods/classes')

    # Initialize total token count
    total_tokens = TOK_COUNT.copy()  

    # Iterate through each custom function
    for i in range(num_custom_funcs):
        # Find the function with the least dependencies
        least_dep_func = min(custom_funcs, key=lambda x: code_dependancies.undocumented_dependancies(x))
        reason = None
        
        # Retry loop for generating documentation
        for ri in range(args.max_retries):
            logging.debug(f'\tTry {ri+1}/{args.max_retries} for `{least_dep_func}`')
            # Generate documentation using a language model
            llm_out, used_toks = get_llm_output(
                SYSTEM_PROMPT,
                DOC_GENERATION_PROMPT(
                    code_dependancies[least_dep_func][CodeData.CODE], 
                    get_reference_docs_custom_functions(least_dep_func, code_dependancies)
                ),
                llm_mode,
                args,
            )
            # Update total tokens used
            total_tokens += used_toks  
            
            # Parse the commented function output from the language model
            new_func_code, new_func_node, success, reason = parse_commented_function(least_dep_func, llm_out)
            
            if not success:
                continue

            # Compare the AST of the original and the new function and check if they match
            same, ast_reason = same_ast_with_reason(remove_docstring(code_dependancies[least_dep_func][CodeData.NODE]), remove_docstring(new_func_node))
            if same:
                # Add the generated code and documentation to code_dependencies
                code_dependancies.add(
                    least_dep_func,
                    {
                        CodeData.CODE_NEW: '\n'.join([code_dependancies[least_dep_func][CodeData.CODE_INDENT] + line for line in new_func_code.split('\n')]),
                        CodeData.DOC: ast.get_docstring(new_func_node),
                    }
                )
                logging.info(f'\t[{str(i+1).zfill(num_digits)}/{str(num_custom_funcs).zfill(num_digits)}] Generated docs for `{least_dep_func}` in {ri+1}/{args.max_retries} tries')      
                break
            else:
                reason = f'AST mismatch: {ast_reason}'
        # Log failure if documentation generation fails after multiple retries
        else:
            logging.info(f'\t[{str(i+1).zfill(num_digits)}/{str(num_custom_funcs).zfill(num_digits)}] Could not generate docs for `{least_dep_func}` after {args.max_retries} tries')
            logging.info(f'\t\tReason: {reason}')
        
        # If documentation was generated, get a shortened version of it
        if code_dependancies[least_dep_func][CodeData.DOC] != '-':
            code_dependancies.add(
                least_dep_func, 
                {CodeData.DOC_SHORT: get_shortened_docs(least_dep_func, CodeData.DOC, args.ref_doc, llm_mode, args)}
            )            

        # Remove the processed function from the list
        custom_funcs.remove(least_dep_func)  
        
    # Generate a list of custom functions that have documentation and Log the results
    custom_funcs_with_docs = [func_name for func_name, func_info in code_dependancies.items() if func_info[CodeData.CUSTOM] and func_info[CodeData.DOC] != '-']
    logging.info(f'Generated docs for {len(custom_funcs_with_docs)}/{num_custom_funcs} custom functions/classes.methods')
    logging.info(f'Tokens used: ' + ', '.join(f'{k}: {v}' for k,v in total_tokens.items()))
    
    
def replace_modified_functions(code_dependancies, path):
    """
    Replaces modified functions in the given code dependencies with new versions.

    Input:
    code_dependancies: dict
        A dictionary containing information about the code dependencies, where keys are function names and values are metadata.
    path: str
        The path where the code dependencies are located.

    Returns:
    None

    Raises:
    IOError: If there is an issue reading from or writing to the file.
    KeyError: If expected keys are not found in the code_dependancies dictionary.
    """
    
    # List comprehension to gather functions with custom implementations and documentation
    # Filters functions that have custom implementations and documentation and stores their names in 'custom_funcs_with_docs' list
    custom_funcs_with_docs = [func_name for func_name, func_info in code_dependancies.items() if func_info[CodeData.CUSTOM] and func_info[CodeData.DOC] != '-']
    # Sort functions with classes appearing first, then functions
    # This ensures that class methods are processed before standalone functions
    custom_funcs_with_docs = sorted(custom_funcs_with_docs, key = lambda func: 1 if code_dependancies[func][CodeData.TYPE] == 'class' else 0)    
    
    # Iterate through the functions that need replacement
    for func in custom_funcs_with_docs: 
        fpath = code_dependancies[func][CodeData.PATH]
        
        # Read the existing file content
        with open(fpath) as f:    
            file_str = f.read()
        
        # Replace the old function code with the new one using replace_func
        file_str = replace_func(
                        func, 
                        code_dependancies[func][CodeData.CODE], 
                        code_dependancies[func][CodeData.CODE_NEW], 
                        fpath,
                        file_str
                    )

        # Write the updated content back to the file
        with open(fpath, 'w') as f:
            f.write(file_str)