import os
import sys
import time
import yaml
import logging
import tempfile
from dotenv import load_dotenv

from evals.base_eval import BaseEval
from llm.inference_coster import CostReport
load_dotenv()
from datetime import datetime, timezone
from typing import List, Tuple
from evals.eval_result import EvalResult
from common import constants as CONST
from models.miner_artifact import Artifact
from db.models.eval import InferenceUsage, db, Miner, Evaluation
from models.eval_type import BitrecsEvaluationType
from evals.eval_factory import EvalFactory
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.responses import FileResponse

app = FastAPI()

logging.basicConfig(level=CONST.LOG_LEVEL)
logger = logging.getLogger(__name__)

logging.getLogger('httpcore').setLevel(logging.WARNING)
logging.getLogger('httpx').setLevel(logging.WARNING)
logging.getLogger('peewee').setLevel(logging.WARNING)
logging.getLogger('fsspec').setLevel(logging.WARNING)



class Actor:
    """Bitrecs Eval Actor"""
    
    def __init__(self):
        self.bitrecs_run_id = os.getenv("BITRECS_RUN_ID", "")
        if not self.bitrecs_run_id:
            raise ValueError("BITRECS_RUN_ID environment variable not set.")      
        logger.info(f"Actor initialized with Bitrecs Run ID: \033[33m{self.bitrecs_run_id}\033[0m")


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
    
    
    def run_eval(self, miner_artifact: Artifact, eval_type: BitrecsEvaluationType) -> Tuple[str, List[EvalResult]]:
        """Run evaluation suites."""      
        run_id = self.bitrecs_run_id
        logger.info(f"Eval Run ID: \033[35m{run_id}\033[0m")
        logger.info(f"TOP RECORDS: \033[36m{CONST.TOP_RECORDS}\033[0m")

        this_set = [eval_type]
        results = EvalFactory.run_all_evals(run_id, miner_artifact, this_set, CONST.TOP_RECORDS)
        
        for result in results:
            print(f"{result}")
            self.log_eval_result_to_db(run_id, result, miner_artifact.miner_hotkey, miner_artifact.model, miner_artifact.provider)
            if result.passed:
                logger.info(f"\033[32m{result.eval_name} Passed! Score: {result.score:.4f}\033[0m")
            else:
                logger.error(f"\033[31m{result.eval_name} Failed! Score: {result.score:.4f}\033[0m")    
        
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


    def get_eval_report(self, run_id: str) -> str:
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
                report_lines.append(f"Sample Size: {eval.rows_evaluated}")
                report_lines.append(f"Notes: {eval.comments}")
                report_lines.append("-" * 60)
            
            report = "\n".join(report_lines)
            #print("\n" + report)
            return report        
        except Exception as e:
            logger.error(f"Failed to generate report for run ID {run_id}: {e}")
        finally:
            db.close()

    
    async def get_inference_report(self, run_id: str) -> dict:    
        try:
            db.connect()
            usage_records = InferenceUsage.select().where(InferenceUsage.run_id == run_id)
            if not usage_records:
                logger.info(f"No inference data found in DB for run ID: {run_id}")
                return {"error": f"No inference data found for run ID: {run_id}"}
            
            data = []
            for usage in usage_records:
                record = {
                    "miner_id": usage.miner_id,
                    "miner_hotkey": usage.hotkey,
                    "model": usage.model,
                    "provider": usage.provider,
                    "prompt_tokens": usage.prompt_tokens,
                    "completion_tokens": usage.completion_tokens,
                    "total_tokens": usage.total_tokens,
                    "finish_reason": usage.finish_reason
                }
                data.append(record)
            
            return {"run_id": run_id, "inference_data": data}
        except Exception as e:
            logger.error(f"Failed to retrieve inference data for run ID {run_id}: {e}")
            return {"error": f"Failed to retrieve inference data for run ID {run_id}"}
        finally:
            db.close()



    async def evaluate(self, yaml_file_path: str, problem_type: BitrecsEvaluationType) -> dict:    
        """
        Evaluation Entrypoint
        """
        bitrecs_run_id = self.bitrecs_run_id
        run_id = None
        try:
            print("=" * 60)
            print("      Bitrecs Evaluation Suite Runner")
            print(f"Local: {datetime.now().isoformat()}")
            print(f"UTC:   {datetime.now(timezone.utc).isoformat()}")
            print("=" * 60)
            start = time.monotonic()        
            
            logger.info(f"Evaluation Type to run: \033[36m{problem_type.value}\033[0m")
            logger.info(f"CONST Max_Iterations per Eval: \033[36m{CONST.TOP_RECORDS}\033[0m")

            #Debug variables:
            # for key, value in os.environ.items():
            #     logger.debug(f"ENV {key}={value}")
            
            logger.info("Loading miner input...")            
            miner_artifact = self.load_miner_input_yaml(input_path=yaml_file_path)
            
            logger.info(f"Artifact ID: {miner_artifact.agent_id}")
            logger.info(f"Artifact Model: {miner_artifact.model}")
            logger.info(f"Artifact Provider: {miner_artifact.provider}")
            logger.info(f"Artifact Hotkey: {miner_artifact.miner_hotkey}")        

            logger.info(f"Starting evaluation: {problem_type.value}")
            run_id, results = self.run_eval(miner_artifact, problem_type)
            logger.info("\033[35mEvaluation completed successfully. \033[0m")
            end = time.monotonic()
            duration = round(end - start, 8)
            
            score = EvalResult.calculate_overall_score(results)

            run_report = self.get_eval_report(run_id)
            logger.info(f"Eval Report for Run ID: \033[35m{run_id}\033[0m")
            logger.info(f"\n{run_report}")

            model_cost_input = float(os.getenv("MODEL_COST_INPUT", 0))
            model_cost_output = float(os.getenv("MODEL_COST_OUTPUT", 0))
            inference_report = await self.get_inference_report(run_id)
            logger.info(f"Inference Report for Run ID: \033[35m{run_id}\033[0m")
            logger.info(f"\n{inference_report}")
            cost_report = CostReport.calculate_cost_from_report(inference_report, 
                                                                input_price_per_million_tokens=model_cost_input, 
                                                                output_price_per_million_tokens=model_cost_output)
            
            token_count = BaseEval.get_run_token_count(run_id)
            
            result = {
                "task_name": problem_type.value,
                "bitrecs_run_id": bitrecs_run_id,
                "run_id": run_id,
                "score": score,
                "success": results[0].passed if results else False,
                "duration": duration,
                "extra": {
                    "result": run_report
                },
                "samples": results[0].rows_evaluated if results else 0,
                "inference_data": inference_report,
                "cost_report": {
                    "input_tokens": cost_report.input_tokens,
                    "output_tokens": cost_report.output_tokens,
                    "total_tokens": token_count,
                    "estimated_cost_usd": cost_report.cost
                }
            }
            
            logger.info(f"Artifact ID: \033[32m{miner_artifact.agent_id}\033[0m")
            logger.info(f"Run ID: \033[33m{bitrecs_run_id}\033[0m")
            logger.info(f"FINAL SCORE \033[92;1m{score:.2f}\033[0m")
            return result
        
        except Exception as e:
            import traceback
            error = f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}"
            logger.error(f"Evaluation failed: {error}")
            end = time.monotonic()
            duration = round(end - start, 8)
            return {
                "task_name": problem_type.value,
                "bitrecs_run_id": bitrecs_run_id,
                "run_id": run_id,
                "score": 0.0,
                "success": False,
                "duration": duration,
                "error": error,
                "error_type": "evaluation_failure",
                "extra": {},
                "samples": 0
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
    problem_name: str

@app.post("/evaluate")
async def evaluate_endpoint(req: EvaluateRequest):
    yaml_content = req.yaml_content    
    actor = Actor()
    env_token = os.getenv("BITRECS_RUN_TOKEN", "")
    #logger.info(f"Env Token: {env_token}, Req Token: {req.run_token}")
    if not env_token or not req.run_token:
        logger.error("Run token not provided.")
        return {"error": "Run token not provided"}
    if env_token != req.run_token:
        logger.error("Invalid run token provided.")
        return {"error": "Invalid run token"}
    MODEL_COST_INPUT = os.getenv("MODEL_COST_INPUT")
    if not MODEL_COST_INPUT:
        logger.error("MODEL_COST_INPUT environment variable not set.")
        return {"error": "MODEL_COST_INPUT environment variable not set."}
    MODEL_COST_OUTPUT = os.getenv("MODEL_COST_OUTPUT")
    if not MODEL_COST_OUTPUT:
        logger.error("MODEL_COST_OUTPUT environment variable not set.")
        return {"error": "MODEL_COST_OUTPUT environment variable not set."}

    eval_type = BitrecsEvaluationType(req.problem_name)        
    try:
        data = yaml.safe_load(yaml_content)
        artifact = Artifact(**data)
        logger.info(f"Miner Hotkey: \033[32m{artifact.miner_hotkey}\033[0m")
        logger.info(f"Artifact ID: \033[32m{artifact.agent_id}\033[0m")
    except Exception as e:
        logger.error(f"Failed to parse yaml into Artifact: {e}")
        return {"error": "Invalid yaml content"}    
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write(yaml_content)
        temp_path = f.name
    try:
        result = await actor.evaluate(temp_path, eval_type)
    finally:
        os.unlink(temp_path)
    return result


@app.get("/run_log/{run_id}")
async def get_run_log(run_id: str) -> dict:
    actor = Actor()
    report = actor.get_eval_report(run_id)
    if not report:
        return {"error": f"No report found for run ID: {run_id}"}
    return {"run_id": run_id, "report": report, "crated_at": datetime.now(timezone.utc).isoformat()}


@app.get("/db")
async def get_db():
    db_path = db.database    
    if not os.path.exists(db_path):
        return {"error": "Database file not found"}    
    
    return FileResponse(
        path=db_path,
        media_type='application/octet-stream',
        filename='eval_runs.db'
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)