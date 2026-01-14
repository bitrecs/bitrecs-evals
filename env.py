import gc
import os
import sys
import time
import yaml
import secrets
import logging
import tempfile
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
from pydantic import BaseModel

app = FastAPI()

logging.basicConfig(level=CONST.LOG_LEVEL)
logger = logging.getLogger(__name__)

logging.getLogger('httpcore').setLevel(logging.DEBUG)
logging.getLogger('httpx').setLevel(logging.DEBUG)
logging.getLogger('peewee').setLevel(logging.DEBUG)


#EVAL_SUITE = ["catalog"]
EVAL_SUITE = ["prompt"]
#EVAL_SUITE = ["catalog", "prompt", "reason"]

class Actor:
    """Bitrecs Eval Actor"""
    
    def __init__(self):
        self.bitrecs_run_id = os.getenv("BITRECS_RUN_ID", "unknown")


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
            raise ValueError(f"Failed to parse YAML into Artifact: {e}")


    def run_eval_suites(self, miner_artifact: Artifact) -> Tuple[str, List[EvalResult]]:
        """Run evaluation suites."""
        logger.info("Running eval suites...")    
        run_id = secrets.token_hex(16)
        logger.info(f"Eval Run ID: \033[35m{run_id}\033[0m")
        logger.info(f"TOP RECORDS: \033[36m{CONST.TOP_RECORDS}\033[0m")
        results = EvalFactory.run_all_evals(run_id, miner_artifact, EVAL_SUITE, CONST.TOP_RECORDS)
        
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
            if not db.is_connection_usable():
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
            if db.is_connection_usable():
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
                report_lines.append(f"Bitrecs Run ID: {self.bitrecs_run_id}")
                report_lines.append(f"Run ID: {eval.run_id}")
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
        bitrecs_run_id = None
        run_id = None
        try:
            print("=" * 60)
            print("      Bitrecs Evaluation Suite Runner")
            print(f"Local: {datetime.now().isoformat()}")
            print(f"UTC:   {datetime.now(timezone.utc).isoformat()}")
            print("=" * 60)
            start = time.monotonic()

            #Debug variables:
            # for key, value in os.environ.items():
            #     logger.debug(f"ENV {key}={value}")
            
            logger.info("Loading miner input...")         
            #miner_input_path = "input/miner_input.yaml"
            miner_artifact = self.load_miner_input_yaml(input_path=yaml_file_path)
            
            logger.info(f"Artifact ID: {miner_artifact.artifact_id}")
            logger.info(f"Model: {miner_artifact.model}")
            logger.info("Starting evaluation suites...")
            logger.info(f"Eval Suites to run: {EVAL_SUITE}, Top Records: {CONST.TOP_RECORDS}")
            run_id, results = self.run_eval_suites(miner_artifact)
            run_report = self.generate_report_by_run_id(run_id)
            logger.info(f"Eval Report for Run ID: \033[35m{run_id}\033[0m")
            logger.info("\n" + run_report)

            logger.info("\033[35mEvaluation suites completed successfully. \033[0m")
            score = sum(r.score for r in results) / len(results) if results else 0.0
            end = time.monotonic()
            durtation = round(end - start, 8)

            bitrecs_run_id = os.getenv("BITRECS_RUN_ID", "unknown")
            logger.info(f"Bitrecs Run ID: \033[33m{bitrecs_run_id}\033[0m")
            result = {
                "task_name": "BitrecsEval",
                "bitrecs_run_id": bitrecs_run_id,
                "run_id": run_id,               
                "score": score,
                "success": score > 0,
                "time_taken": durtation,
                "extra": {
                    "result": run_report
                }
            }            
            return result
        
        except Exception as e:
            import traceback
            error = f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}"
            logger.error(f"Evaluation failed: {error}")
            end = time.monotonic()
            durtation = round(end - start, 8)         
            return {
                "task_name": "BitrecsEval",
                "bitrecs_run_id": bitrecs_run_id,
                "run_id": run_id,
                "score": 0.0,
                "success": False,
                "time_taken": durtation,
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

class EvaluateRequest(BaseModel):
    yaml_content: str
    run_token: str

@app.post("/evaluate")
async def evaluate_endpoint(req: EvaluateRequest):
    yaml_content = req.yaml_content    
    actor = Actor()
    env_token = os.getenv("BITRECS_RUN_TOKEN", "")
    if not env_token or not req.run_token:
        logger.error("Run token not provided.")
        return {"error": "Run token not provided"}
    if env_token != req.run_token:
        logger.error("Invalid run token provided.")
        return {"error": "Invalid run token"}
        
    try:
        data = yaml.safe_load(yaml_content)
        artifact = Artifact(**data)
        logger.info(f"Miner Hotkey: \033[32m{artifact.miner_hotkey}\033[0m")
    except Exception as e:
        logger.error(f"Failed to parse yaml into Artifact: {e}")
        return {"error": "Invalid yaml content"}    
    
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