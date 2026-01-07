import gc
import os
import sys
import time
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
from fastapi import FastAPI

app = FastAPI()

logging.basicConfig(level=CONST.LOG_LEVEL)
logger = logging.getLogger(__name__)

logging.getLogger('httpcore').setLevel(logging.WARNING)
logging.getLogger('httpx').setLevel(logging.WARNING)
logging.getLogger('peewee').setLevel(logging.WARNING)

TOP_RECORDS = 3
#EVAL_SUITE = ["catalog"]
EVAL_SUITE = ["prompt"]
#EVAL_SUITE = ["catalog", "prompt", "reason"]

class Actor:
    """Bitrecs Eval Actor"""
    
    def __init__(
        self,
        api_key: str = None,
    ):
        """
        Initialize Actor with API key
        
        Args:
            api_key: API key for LLM service. If not provided, will use OPENROUTER_API_KEY env var
        """
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        
        # Initialize trace task instance
        #self.trace_task = TraceTask()


    def load_miner_input_yaml(self, input_path=None) -> Artifact:
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


    def run_eval_suites(self, miner_artifact: Artifact) -> Tuple[str, List[EvalResult]]:
        """Run evaluation suites."""
        logger.info("Running eval suites...")    
        run_id = secrets.token_hex(16)
        logger.info(f"Eval Run ID: \033[35m{run_id}\033[0m")
        logger.info(f"TOP RECORDS: \033[36m{TOP_RECORDS}\033[0m")
        results = EvalFactory.run_all_evals(run_id, miner_artifact, EVAL_SUITE, TOP_RECORDS)
        
        for result in results:
            print(f"{result}")
            self.log_eval_result_to_db(run_id, result, miner_artifact.miner_hotkey, miner_artifact.model, miner_artifact.provider)
            if result.passed:
                logger.info(f"\033[32m{result.eval_name} Passed! Score: {result.score:.2f}\033[0m")
            else:
                logger.error(f"\033[31m{result.eval_name} Failed! Score: {result.score:.2f}\033[0m")    
        
        total_score = sum(r.score for r in results) / len(results) if results else 0.0
        logger.info(f"Aggregated Score: {total_score:.2f}")        
        logger.info(f"RUN COMPLETE for run ID: \033[34m{run_id}\033[0m")
        return run_id, results


    def log_eval_result_to_db(self, run_id: str, result: EvalResult, hotkey, model_name, provider_name):
        """Log EvalResult to the database."""
        try:
            db.connect()        
            #db.drop_tables([Miner, Evaluation], safe=True)
            db.create_tables([Miner, Evaluation], safe=True)  # Ensure tables exist

            # Get or create Miner
            miner, created = Miner.get_or_create(hotkey=hotkey)

            # Create Evaluation record
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
            db.close()


    def display_eval_results(self):
        """Display eval results from the database."""
        try:
            db.connect()
            evaluations = Evaluation.select().order_by(Evaluation.created_at.desc()).limit(10)  # Last 10 results
            if not evaluations:
                logger.info("No eval results found in DB.")
                return
            
            print("\n" + "=" * 60)
            print("      Recent Eval Results")
            print("=" * 60)
            for eval in evaluations:
                print(f"ID: {eval.id} | Eval: {eval.eval_name} | Model: {eval.model_name} | Provider: {eval.provider_name} | Score: {eval.score:.2f} | Success: {eval.success} | Duration: {eval.duration_seconds:.2f}s | Success Rows: {eval.rows_evaluated} | Comments: {eval.comments}")
            print("=" * 60)
        except Exception as e:
            logger.error(f"Failed to display results: {e}")
        finally:
            db.close()


    def display_eval_results_by_run_id(self, run_id: str):
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

    def generate_report_by_run_id(self, run_id: str) -> str:
        """Generate a detailed report for a specific run ID."""
        try:
            db.connect()
            evaluations = Evaluation.select().where(Evaluation.run_id == run_id)
            if not evaluations:
                logger.info(f"No eval results found in DB for run ID: {run_id}")
                return
            
            report_lines = []
            report_lines.append(f"Eval Report for Run ID: {run_id}")
            report_lines.append("=" * 60)
            for eval in evaluations:
                report_lines.append(f"Eval: {eval.eval_name}")
                report_lines.append(f"Model: {eval.model_name}")
                report_lines.append(f"Provider: {eval.provider_name}")
                report_lines.append(f"Score: {eval.score:.2f}")
                report_lines.append(f"Success: {eval.success}")
                report_lines.append(f"Duration: {eval.duration_seconds:.2f}s")
                report_lines.append(f"Rows Evaluated: {eval.rows_evaluated}")
                report_lines.append(f"Comments: {eval.comments}")
                report_lines.append("-" * 60)
            
            report = "\n".join(report_lines)
            #print("\n" + report)
            return report        
        except Exception as e:
            logger.error(f"Failed to generate report for run ID {run_id}: {e}")
        finally:
            db.close()

    async def evaluate(self, yaml_file_path: str) -> dict:    
        """
        Affine Entrypoint
        """
        try:
            print("=" * 60)
            print("      Bitrecs Evaluation Suite Runner")
            print(f"Local: {datetime.now().isoformat()}")
            print(f"UTC:   {datetime.now(timezone.utc).isoformat()}")
            print("=" * 60)
            start = time.monotonic()
            
            logger.info("Loading miner input...")
            with open(yaml_file_path, 'r') as f:
                data = yaml.safe_load(f)
        
            # model = data.get("model")
            # base_url = data.get("base_url")
            # task_id = data.get("task_id")
            # miner_hotkey = data.get("miner_hotkey")        
            #miner_input_path = "input/miner_input.yaml"
            miner_artifact = self.load_miner_input_yaml(input_path=yaml_file_path)
            
            # # Override with provided params if given
            # if model:
            #     miner_artifact.model = model
            # if base_url:
            #     # Assuming base_url indicates provider; adjust as needed
            #     miner_artifact.provider = "chutes" if "chutes" in base_url else miner_artifact.provider
            # if task_id:
            #     logger.info(f"Task ID: {task_id}")
            
            logger.info(f"Artifact ID: {miner_artifact.artifact_id}")
            logger.info(f"Model: {miner_artifact.model}")
            logger.info("Starting evaluation suites...")
            run_id, results = self.run_eval_suites(miner_artifact)
            run_report = self.generate_report_by_run_id(run_id)
            logger.info(f"Eval Report for Run ID: \033[35m{run_id}\033[0m")
            logger.info("\n" + run_report)

            logger.info("\033[35mEvaluation suites completed successfully. \033[0m")
            score = sum(r.score for r in results) / len(results) if results else 0.0
            result = {
                "task_name": "Trace",
                "score": score,
                "success": score > 0,
                "time_taken": time.time() - start,
                "extra": {
                    "result": run_report
                }
            }        
        
            gc.collect()
            return result
        
        except Exception as e:
            import traceback
            error = f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}"
            logger.error(f"Evaluation failed: {error}")
            return {
                "task_name": "Trace",
                "score": 0.0,
                "success": False,
                "time_taken": 0.0,
                "error": error,
                "error_type": "evaluation_failure",
                "extra": {}
            }

@app.get("/")
async def root():
    return {"message": "Bitrecs Eval Actor is running."}

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

@app.post("/evaluate")
async def evaluate_endpoint(yaml_content: str):
    api_key = os.getenv("OPENROUTER_API_KEY")
    actor = Actor(api_key=api_key)
    # Load the yaml content into a dict
    data = yaml.safe_load(yaml_content)
    # Create artifact from the loaded data
    try:
        artifact = Artifact(**data)
        logger.info(f"Miner Hotkey: \033[32m{artifact.miner_hotkey}\033[0m")
    except Exception as e:
        logger.error(f"Failed to parse yaml into Artifact: {e}")
        return {"error": "Invalid yaml content"}
    
    # Since evaluate expects a file path, save to a temp file
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write(yaml_content)
        temp_path = f.name
    try:
        result = await actor.evaluate(temp_path)
    finally:
        os.unlink(temp_path)
    return result

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8081)