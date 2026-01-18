import os
import re
import sys
import yaml
import secrets
import logging
from dotenv import load_dotenv
load_dotenv()
from datetime import datetime, timezone
from typing import List, Tuple
from evals.eval_result import EvalResult
from common import constants as CONST
from models.miner_artifact import Artifact
from db.models.eval import db, Miner, Evaluation
from evals.eval_factory import EvalFactory
from models.eval_type import BitrecsEvaluationType

logging.basicConfig(level=CONST.LOG_LEVEL)
logger = logging.getLogger(__name__)

logging.getLogger('httpcore').setLevel(logging.WARNING)
logging.getLogger('httpx').setLevel(logging.WARNING)
logging.getLogger('peewee').setLevel(logging.WARNING)


#$EVAL_SUITE = [BitrecsEvaluationType.BITRECS_BASIC_DAILY, BitrecsEvaluationType.BITRECS_REASON_DAILY]
# EVAL_SUITE = [BitrecsEvaluationType.BITRECS_BASIC_DAILY, 
#               BitrecsEvaluationType.BITRECS_REASON_DAILY, 
#               BitrecsEvaluationType.BITRECS_SKU_DAILY,
#               BitrecsEvaluationType.BITRECS_PROMPT_DAILY]

#EVAL_SUITE = [BitrecsEvaluationType.BITRECS_BASIC_DAILY, BitrecsEvaluationType.BITRECS_REASON_DAILY, BitrecsEvaluationType.BITRECS_PROMPT_DAILY]

EVAL_SUITE = [BitrecsEvaluationType.BITRECS_BASIC_DAILY, 
              BitrecsEvaluationType.BITRECS_PROMPT_DAILY, ]
              


def load_miner_input_yaml(input_path=None) -> Artifact:
    """Load miner input YAML and convert to Artifact object."""    
    if not input_path or not os.path.exists(input_path):
        logger.error(f"Error: {input_path} not found.")
        sys.exit(1)
    logger.info(f"Loaded miner input from \033[32m{input_path}\033[0m")    
    with open(input_path, 'r') as f:
        data = yaml.safe_load(f)
    try:
        artifact = Artifact(**data)
        logger.info(f"Miner Hotkey: \033[32m{artifact.miner_hotkey}\033[0m")
        return artifact
    except Exception as e:
        logger.error(f"Failed to parse miner input into Artifact: {e}")
        sys.exit(1)


def run_eval_suites(miner_artifact: Artifact, shuffle=False) -> Tuple[str, List[EvalResult]]:
    """Run evaluation suites."""
    logger.info("Running eval suites...")    
    run_id = f"test_{secrets.token_hex(16)}"
    logger.info(f"Eval Run ID: \033[35m{run_id}\033[0m")
    results = EvalFactory.run_all_evals(run_id, miner_artifact, EVAL_SUITE, CONST.TOP_RECORDS)
    
    for result in results:
        #print(f"{result}")
        log_eval_result_to_db(run_id, result, miner_artifact.miner_hotkey, miner_artifact.model, miner_artifact.provider)
        if result.passed:
            logger.info(f"\033[32m{result.eval_name} Passed! Score: {result.score:.2f}\033[0m")
        else:
            logger.error(f"\033[31m{result.eval_name} Failed! Score: {result.score:.2f}\033[0m")    
    
    total_score = sum(r.score for r in results) / len(results) if results else 0.0
    
    logger.info(f"Aggregated Score: {total_score:.2f} from {len(results)} evals.")    
    logger.info(f"RUN COMPLETE for run ID: \033[34m{run_id}\033[0m")    
    return run_id, results


def log_eval_result_to_db(run_id: str, result: EvalResult, hotkey, model_name, provider_name):
    """Log EvalResult to the database."""
    try:
        if not db.is_connection_usable():
            db.connect()
        db.create_tables([Miner, Evaluation], safe=True)
        miner, created = Miner.get_or_create(hotkey=hotkey)        
        Evaluation.create(
            run_id=run_id,
            miner=miner,
            eval_name=result.eval_name,
            model_name=model_name,
            provider_name=provider_name,
            score=result.score,
            success=result.passed,
            duration_seconds=result.duration_seconds,
            comments=result.details,
            rows_evaluated=result.rows_evaluated
        )
        logger.info("Eval result logged to DB.")
    except Exception as e:
        logger.error(f"Failed to log to DB: {e}")
    finally:
        if db.is_connection_usable():
            db.close()


def display_eval_results_by_run_id(run_id: str):
    """Display eval results for a specific run ID from the database."""
    try:
        db.connect()
        evaluations = Evaluation.select().where(Evaluation.run_id == run_id)
        if not evaluations:
            logger.info(f"No eval results found in DB for run ID: {run_id}")
            return
        
        print("\n" + "=" * 60)
        print(f"      Eval Results for Run ID: {run_id}")
        print("=" * 60)
        for eval in evaluations:
            print(f"ID: {eval.id} | Eval: {eval.eval_name} | Model: {eval.model_name} | Provider: {eval.provider_name} | Score: {eval.score:.2f} | Success: {eval.success} | Duration: {eval.duration_seconds:.2f}s | Success Rows: {eval.rows_evaluated} | Comments: {eval.comments}")
        print("=" * 60)
    except Exception as e:
        logger.error(f"Failed to display results for run ID {run_id}: {e}")
    finally:
        db.close()

def strip_ansi(text: str) -> str:
    """Strip ANSI escape sequences from text."""
    ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
    return ansi_escape.sub('', text)

def generate_report_by_run_id(run_id: str) -> str:
    """Generate a detailed report for a specific run ID."""
    try:
        db.connect()
        evaluations = Evaluation.select().where(Evaluation.run_id == run_id)
        if not evaluations:
            logger.info(f"No eval results found in DB for run ID: {run_id}")
            return ""
        
        report_lines = []
        report_lines.append(f"Eval Report for Run ID: {run_id}")
        report_lines.append("=" * 60)
        for eval in evaluations:
            report_lines.append(f"Eval: {eval.eval_name}")
            if eval.success:
                report_lines.append("Result:\033[32m PASS\033[0m")                
            else:
                report_lines.append("Result:\033[31m FAIL\033[0m")

            report_lines.append(f"Sample Size: {eval.rows_evaluated}")
            report_lines.append(f"Provider: \033[33m{eval.provider_name}\033[0m")
            report_lines.append(f"Model: \033[36m{eval.model_name}\033[0m")
            report_lines.append(f"Duration: {eval.duration_seconds:.2f}s")
            report_lines.append(f"Score: {eval.score:.2f}")
            report_lines.append(f"Comments: {eval.comments}")
            report_lines.append("-" * 60)
        
        report = "\n".join(report_lines)                
        return report        
    except Exception as e:
        logger.error(f"Failed to generate report for run ID {run_id}: {e}")
    finally:
        db.close()

def write_log_to_output_file(log_content: str, output_path: str):
    """Write log content to an output file."""
    try:
        with open(output_path, 'w') as f:
            f.write(log_content)
        logger.info(f"Log written to \033[32m{output_path}\033[0m")
    except Exception as e:
        logger.error(f"Failed to write log to {output_path}: {e}")


def main():
    print("=" * 60)
    print("      Bitrecs Evaluation Suite Runner")
    print(f"Local: {datetime.now().isoformat()}")
    print(f"UTC:   {datetime.now(timezone.utc).isoformat()}")
    print("=" * 60)
    
    logger.info("Loading miner input...")
    miner_input_path = "input/miner_input.yaml"
    miner_artifact = load_miner_input_yaml(input_path=miner_input_path)
    
    logger.info(f"Artifact ID: {miner_artifact.artifact_id}")
    logger.info(f"Model: {miner_artifact.model}")
    logger.info("Starting evaluation suites...")
    logger.info(f"Eval Suites to run: {EVAL_SUITE}, Top Records: {CONST.TOP_RECORDS}")
    run_id, results = run_eval_suites(miner_artifact)    

    run_report = generate_report_by_run_id(run_id) or ""
    logger.info(f"Eval Report for Run ID: \033[35m{run_id}\033[0m")
    logger.info("\n" + run_report)
    logger.info("\033[35mEvaluation suites completed successfully. \033[0m")
    run_log = strip_ansi(run_report)    
    write_log_to_output_file(run_log, output_path=f"output/eval_report_{run_id}.txt")

    final_score = EvalResult.calculate_overall_score(results)
    logger.info(f"\033[34mFinal Overall Score: {final_score:.2f}\033[0m")

    


if __name__ == "__main__":
    main()