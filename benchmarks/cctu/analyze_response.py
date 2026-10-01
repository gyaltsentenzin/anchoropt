import argparse
import json

def print_codes(all_code, code_str):
    codes_dict = json.loads(code_str)
    for function_name, code in codes_dict.items():
        if function_name not in all_code:
            all_code.append(function_name)
            print(code)
            print("\n\n\n")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default="granite-4-1-8b", help='Model name: API models ')
    parser.add_argument('--split', type=str, default="train", choices=["all", "train", "test"])
    # The run is identified by its POLICY, not by a hand-incremented counter. `--controllers` names
    # the same spec file the run used, so this resolves to the same directory response_generator wrote
    # to; `--trial` is gone because it only renamed directories and was never read (see cctu_paths).
    parser.add_argument('--controllers', type=str, default=None,
                        help='the controller spec the run used. Omitted = the baseline.')
    # Runs are kept under a per-MODEL root so two models' outputs cannot be differenced by accident
    # -- anchors are mined from one model's residual and do not transfer as artifacts.
    parser.add_argument('--results-root', type=str, default="results",
                        help='root holding the per-model run directories (default: results)')
    args = parser.parse_args()

    # THE SAME derivation response_generator.py and run_hosted.py use. One owner, so a reader cannot
    # point this at a directory the run never wrote to.
    import cctu_paths
    run_dir = cctu_paths.run_dir(root=args.results_root, model=args.model, split=args.split,
                                 controllers=args.controllers)
    output_file = cctu_paths.artifact(run_dir, "analysis.jsonl")
    with open(output_file, "r") as f:
        all_samples = [json.loads(line) for line in f.readlines()]
    all_code = []

    # analyze each sample
    for sample in all_samples:
        data_source = sample['analysis_entry']['sample']['data_source']
        if data_source != "Single-Hop":
            continue
        id = sample["id"]
        print('\n\n\n', '-'*10, f"sample id {id}", '-'*10)

        # sample data
        system_prompt = sample['analysis_entry']['sample']['messages'][0]['content']
        user_prompt = sample['analysis_entry']['sample']['messages'][1]['content']
        query = user_prompt.split('Question:')[-1].strip()
        print(f"Query: {query}")

        unsolved_set = sample['analysis_entry']['sample']['unsolved_set']
        answer = sample['analysis_entry']['sample']['answer']
        print(f"Answer: {answer}")

        tool_str = sample['analysis_entry']['sample']['tools']
        code_str = sample['analysis_entry']['sample']['codes']
        constraints_list = sample['analysis_entry']['sample']['constraints_list']
        # print_codes(all_code, code_str)

        # conversation
        conversation = sample['analysis_entry']['conversation']
        for turn in conversation:
            thought = turn['assistant_msg']['content']
            if thought:
                print(f'\tAssistant: {thought}')
            tool_to_be_called_list = turn['assistant_msg']['tool_calls']
            for i in range(len(tool_to_be_called_list)):
                function = tool_to_be_called_list[i]['function']['name']
                arguments = tool_to_be_called_list[i]['function']['arguments']
                feedback = turn['feedback'][i]
                print(f"\tCalling {function} function")
                print(f"\tArguments {arguments}")
                print(f"\tReturn {feedback}\n")

if __name__ == '__main__':
    main()