import ast
import hashlib
import os
import json
import re
import sqlite3
import string
import pandas as pd
from typing import List, Tuple, Union
from dataclasses import dataclass
from pandas import DataFrame
#from batch_report_presenter import BatchReportPresenter
from datetime import datetime, timedelta, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed
from commerce.product_factory import CatalogProvider, ProductFactory


MIN_REQUESTS = 15  
SKU_DUPLICATION_THRESHOLD = 0.40

@dataclass
class ReasonedProduct:
    sku : str
    name : str
    price : str
    reason: str 
    
    @staticmethod
    def from_json(json_str: str) -> List['ReasonedProduct']:
        reasoned_products = []
        try:            
            items = json.loads(json_str)
        except json.JSONDecodeError:
            try:                
                items = ast.literal_eval(json_str)
            except (ValueError, SyntaxError) as e:
                print(f"Failed to parse data: {e}")
                return reasoned_products    
        
        for item in items:
            if isinstance(item, dict):                
                data = item
            elif isinstance(item, str):                
                try:
                    data = json.loads(item)
                except json.JSONDecodeError:
                    try:
                        data = ast.literal_eval(item)
                    except:
                        print(f"Failed to parse item: {item}")
                        continue
            else:
                print(f"Unexpected item type: {type(item)}")
                continue            
            
            if isinstance(data, dict):
                rp = ReasonedProduct(
                    sku=data.get("sku", ""),
                    name=data.get("name", ""),
                    price=data.get("price", ""),
                    reason=data.get("reason", "")
                )
                reasoned_products.append(rp)
    
        return reasoned_products
    


class MinerReport:
    created_at: str
    scored_at: str
    miner_hotkey: str
    miner_uid: str
    miner_ip: str
    num_success: int
    num_failures: int
    
    num_requests_evaluated: int
    total_unique_products: int
    avg_response_time: float

    r_score: float # Reasoning Score
    s_score: float # SKU Relevance Score
    f_score: float # Final Score

    report_card: str
    models_used: List[str] = []    
    evalator_notes: List[str] = []
    rank: int = -1    

    def to_dict(self):
        return {
            'created_at': self.created_at,
            'scored_at': self.scored_at,
            'miner_hotkey': self.miner_hotkey,
            'miner_uid': self.miner_uid,
            'miner_ip': self.miner_ip,
            'num_success': self.num_success,
            'num_failures': self.num_failures,
            'num_requests_evaluated': self.num_requests_evaluated,
            'total_unique_products': self.total_unique_products,
            'avg_response_time': self.avg_response_time,
            'r_score': self.r_score,
            's_score': self.s_score,
            'f_score': self.f_score,
            'report_card': self.report_card,
            'models_used': self.models_used,
            'evalator_notes': self.evalator_notes,
            'rank': self.rank
        }


class BatchReport:    
    batch_id: str
    batch_date: str
    query: str
    avg_axon_process_time: float 
    avg_dendrite_process_time: float
    num_results: int
    batch_elected_miner_uid: str    
    batch_elected_miner_hotkey: str = ""
    batch_elected_model: str = ""
    batch_elected_result: str = ""
    keys: List[str]
    scores: List[float]
    axon_status_codes: List[str]
    dendrite_status_codes: List[str]   

    def to_dict(self):
        return {
            'batch_id': self.batch_id,
            'batch_date': self.batch_date,
            'query': self.query,
            'avg_axon_process_time': self.avg_axon_process_time,
            'avg_dendrite_process_time': self.avg_dendrite_process_time,
            'num_results': self.num_results,
            'batch_elected_miner_uid': self.batch_elected_miner_uid,
            'batch_elected_miner_hotkey': self.batch_elected_miner_hotkey,
            'batch_elected_model': self.batch_elected_model,
            'batch_elected_result': self.batch_elected_result,
            'keys': self.keys,
            'scores': self.scores,
            'axon_status_codes': self.axon_status_codes,
            'dendrite_status_codes': self.dendrite_status_codes
        }
    


class RulesScorer:
    def __init__(self, db_full_path: str, max_workers: int = 4, debug: bool = False):
        self.db_dir = db_full_path
        self.debug = debug
        db_files = []       
        db_files.append(db_full_path)
        
        if len(db_files) == 0:
            raise FileNotFoundError("No SQLite database files found in the specified directory")
        
        woo_products = ProductFactory.load_default_catalog(CatalogProvider.WOO_COMMERCE)
        self.woo_catalog = tuple(woo_products)  # Convert to tuple for read-only access
        
        self.db_files = db_files
        self.max_workers = max_workers
        indexs_updated = self.init_indicies()
        print(f"Database indices updated: {indexs_updated}")

        bad_hotkeys = self.get_verified_proof_abusive_hotkeys(days_back=14, error_threshold=0.80)
        self.abusive_keys = bad_hotkeys if bad_hotkeys else []

        self.terrible_reasons = {            
            "good", "great", "nice", "ok", "fine", "recommended", "popular",
            "best", "top", "excellent", "amazing", "awesome", "perfect",
            "cool", "neat", "lovely", "wonderful", "fantastic", "incredible",
            "outstanding", "superb", "brilliant", "stellar", "epic", "solid",
            "sweet", "sick", "fire", "lit", "dope", "rad", "bad", "terrible",
            "awful", "horrible", "cheap", "expensive", "useful", "helpful",            
            
            "very good", "really good", "so good", "pretty good", "quite good",
            "very nice", "really nice", "so nice", "pretty nice", "quite nice",
            "very cool", "really cool", "so cool", "pretty cool", "quite cool",
            "really great", "very great", "so great", "pretty great",
            "this is good", "this is great", "this is nice", "this is cool",
            "this is awesome", "this is amazing", "this is excellent",
            "this is perfect", "this is wonderful", "this is fantastic",
            "highly recommended", "strongly recommended", "definitely recommended",
            "would recommend", "i recommend", "recommend this", "recommend it",            
            
            "high quality", "good quality", "great quality", "excellent quality",
            "top quality", "premium quality", "best quality", "superior quality",            
            
            "buy this", "buy it", "get this", "get it", "purchase this",
            "worth it", "not worth it", "good buy", "bad buy", "smart buy",            
            
            "works", "works well", "works great", "doesnt work", "broken",
            "love it", "hate it", "like it", "dislike it", "enjoy it",
            "satisfied", "happy", "pleased", "disappointed", "unhappy",            
            
            "as described", "exactly as described", "just as described",
            "what i expected", "as expected", "what you expect", "expected quality"
        }

    def init_indicies(self) -> bool:
        db_file = self.db_dir
        conn = sqlite3.connect(db_file)
        cursor = conn.cursor()
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_created_at ON miner_responses(created_at)")
        conn.commit()
        #print(f"Created index on created_at in {db_file}")

        cursor.execute("CREATE INDEX IF NOT EXISTS idx_bt_header_axon_hotkey ON miner_responses(bt_header_axon_hotkey)")
        conn.commit()
        #print(f"Created index on idx_bt_header_axon_hotkey in {db_file}")
        
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_site_key ON miner_responses(site_key)")
        conn.commit()
        #print(f"Created index on idx_site_key in {db_file}")
        
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_miner_uid ON miner_responses(miner_uid)")
        conn.commit()
        #print(f"Created index on idx_miner_uid in {db_file}")

        cursor.execute("CREATE INDEX IF NOT EXISTS idx_miner_hotkey ON miner_responses(miner_hotkey)")
        conn.commit()
        #print(f"Created index on idx_miner_hotkey in {db_file}")
        
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_query ON miner_responses(query)")
        conn.commit()
        #print(f"Created index on idx_query in {db_file}")
        
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_status_code ON miner_responses(bt_header_axon_status_code )")
        conn.commit()
        #print(f"Created index on idx_status_code in {db_file}")
        print("Checked / Created indices in {}".format(db_file))
        return True       
        

    def get_stats(self) -> dict:
        stats = {
            'db_dir': self.db_dir,
            'db_files': self.db_files,
            'max_workers': self.max_workers,
            'debug': self.debug,
            'num_db_files': len(self.db_files),
            'catalog_size': len(self.woo_catalog)
        }
        return stats
    
   
    def get_batch_ids(self) -> List[str]:
        batch_ids = set()
        for db_file in self.db_files:
            #conn = sqlite3.connect(db_file)
            conn = sqlite3.connect(f"file:{db_file}?mode=ro", uri=True)
            df = pd.read_sql_query("SELECT DISTINCT site_key FROM miner_responses", conn)
            batch_ids.update(df['site_key'].tolist())
            conn.close()        
        return list(batch_ids)
    
    def get_batch_ids_since(self, since_date: datetime) -> List[str]:
            batch_ids = set()
            since_dt = since_date.strftime('%Y-%m-%d %H:%M:%S')
            for db_file in self.db_files:
                #conn = sqlite3.connect(db_file)
                conn = sqlite3.connect(f"file:{db_file}?mode=ro", uri=True)
                df = pd.read_sql_query(
                    "SELECT DISTINCT site_key FROM miner_responses WHERE created_at > ?",
                    conn,
                    params=(since_dt,)
                )
                batch_ids.update(df['site_key'].tolist())
                conn.close()
            return list(batch_ids)
    
    def get_dataframe_by_batch(self, batch_id: str) -> Union[pd.DataFrame, None]:
        final_df = None
        for db_file in self.db_files:
            #conn = sqlite3.connect(db_file)
            conn = sqlite3.connect(f"file:{db_file}?mode=ro", uri=True)
            df = pd.read_sql_query("SELECT * FROM miner_responses WHERE site_key=?", conn, params=(batch_id,))
            print(f"loaded db from {db_file} for batch {batch_id}")
            if final_df is None:
                final_df = df
            else:
                final_df = pd.concat([final_df, df], ignore_index=True) 
            conn.close()        
        return final_df
    
    def get_dataframe_by_miner(self, miner_hotkey: str) -> Union[pd.DataFrame, None]:
        final_df = None
        for db_file in self.db_files:
            #conn = sqlite3.connect(db_file)
            conn = sqlite3.connect(f"file:{db_file}?mode=ro", uri=True)
            df = pd.read_sql_query("SELECT * FROM miner_responses WHERE bt_header_axon_hotkey=?", conn, params=(miner_hotkey,))
            print(f"loaded db from {db_file} for miner {miner_hotkey}")
            if final_df is None:
                final_df = df
            else:
                final_df = pd.concat([final_df, df], ignore_index=True) 
            conn.close()
        return final_df
    

    def is_suspect_miner(self, miner_hotkey: str) -> bool:
        """
        Check if the miner has made requests from known abusive IP patterns in the last 30 days.
        """
        return self.abusive_keys and miner_hotkey in self.abusive_keys
    
    
    def get_verified_proof_abusive_hotkeys(self, days_back: int = 7, error_threshold: float = 0.80) -> List[str]:
        vi_since = (datetime.now(timezone.utc) - timedelta(days=days_back)).strftime('%Y-%m-%d %H:%M')
        vi_proof = self.get_verified_proof_table(since_date=vi_since)
        if vi_proof.empty:
            print("No verified proof data found.")
            return []        
        filtered_vi_proof = vi_proof[vi_proof['success_rate_float'] < error_threshold]        
        vi_fakers = filtered_vi_proof["miner"].unique().tolist()        
        return vi_fakers
            

    def get_verified_proof_table(self, since_date: str = '2025-10-24') -> pd.DataFrame:
        """
        Runs the verified proof table query and returns a DataFrame.
        The query analyzes miner responses with verified proofs since the given date.
        """
        sql = f"""        
        WITH ranked AS (
            SELECT
                bt_header_axon_hotkey,
                bt_header_axon_ip,
                verified_proof,
                reward,
                created_at,
                ROW_NUMBER() OVER (
                    PARTITION BY bt_header_axon_hotkey 
                    ORDER BY created_at DESC
                ) AS rn
            FROM miner_responses
            WHERE created_at > '{since_date}'
            AND verified_proof IS NOT NULL 
            AND verified_proof != ''
            AND bt_header_axon_ip IS NOT NULL 
            AND bt_header_axon_ip != ''
        ),
        stats AS (
            SELECT
                bt_header_axon_hotkey AS miner,
                MAX(CASE WHEN rn = 1 THEN bt_header_axon_ip END) AS ip,
                COUNT(*) AS total_requests,
                COUNT(*) AS verified_proof_count,
                SUM(CASE WHEN CAST(reward AS REAL) > 0 THEN 1 ELSE 0 END) AS success_count,
                SUM(CASE WHEN CAST(reward AS REAL) = 0 OR reward IS NULL THEN 1 ELSE 0 END) AS fail_with_proof_count,
                0 AS fail_without_proof_count
            FROM ranked
            GROUP BY bt_header_axon_hotkey
            HAVING success_count < verified_proof_count - 1
        )
        SELECT
            miner,
            ip,
            total_requests,
            verified_proof_count,
            success_count,
            fail_with_proof_count,
            fail_without_proof_count,
            printf("%.2f", 100.0 * success_count * 1.0 / verified_proof_count) || '%' AS success_rate_pct,
            CAST(success_count AS REAL) / verified_proof_count AS success_rate_float
        FROM stats
        ORDER BY (total_requests - success_count) DESC;
        """
        
        final_df = None
        for db_file in self.db_files:
            conn = sqlite3.connect(f"file:{db_file}?mode=ro", uri=True)
            df = pd.read_sql_query(sql, conn)
            if final_df is None:
                final_df = df
            else:
                final_df = pd.concat([final_df, df], ignore_index=True)
            conn.close()
        
        return final_df if final_df is not None else pd.DataFrame()
    
    
    def score_all_miners_since_parallel(self, days_ago: int = 7, top: int = 256, min_success: int = 50) -> List[MinerReport]:
        print("Scoring all miners since {} days ago with {} max workers enabled ".format(days_ago, self.max_workers))
        reports = []
        since_date = datetime.now() - timedelta(days=days_ago)
        print(f"Scoring all miners since {since_date.strftime('%Y-%m-%d %H:%M:%S')}")
        final_df = None
        for db_file in self.db_files:
            conn = sqlite3.connect(db_file)
            df = pd.read_sql_query(
                "select DISTINCT bt_header_axon_hotkey from miner_responses where created_at > ?",
                conn, params=(since_date.strftime('%Y-%m-%d %H:%M:%S'),))
            if final_df is None:
                final_df = df
            else:
                final_df = pd.concat([final_df, df], ignore_index=True)
            conn.close()

        if final_df is None or final_df.empty:
            print(f"No miners found since {since_date}")
            return []

        miner_hotkeys = final_df['bt_header_axon_hotkey'].unique()
        if top is not None and len(miner_hotkeys) > top:
            print(f"Capping to top {top} miners for scoring.")
            miner_hotkeys = miner_hotkeys[:top]        

        print(f"Found \033[32m{len(miner_hotkeys)} miners to score since {since_date.strftime('%Y-%m-%d %H:%M:%S')} \033[0m")
        if len(miner_hotkeys) > 256:
            print(f"⚠️ Too many miners to score: {len(miner_hotkeys)}. Consider reducing the time window or increasing the top N limit.")
            #print(f"Too many miners to score: {len(miner_hotkeys)}. Limiting to 250.")
            #miner_hotkeys = miner_hotkeys[:250]
            #return         

        print(f"Threshold for SKU duplication across queries: {SKU_DUPLICATION_THRESHOLD:.0%}")
        print(f"Minimum requests per miner to be scored: {MIN_REQUESTS}")
        print(f"Starting parallel scoring using {self.max_workers} workers...")
        
        #miner_hotkeys = ["5CyQjHzKLYiQNCsLTJE97gV8TbQeGGJCLNCfsLZS7NMSAA49"]
        # Parallel scoring
        workers = self.max_workers
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(self.score_miner, miner_hotkey, days_ago, min_success): miner_hotkey for miner_hotkey in miner_hotkeys}
            for future in as_completed(futures):
                print("finished scoring miner")                
                report = future.result()
                if report:
                    reports.append(report)

        reports : List[MinerReport] = [r for r in reports if r is not None and r.num_requests_evaluated >= MIN_REQUESTS]
        skipped_reports = [r for r in reports if r is None or r.num_requests_evaluated < MIN_REQUESTS]
        reports.sort(
            key=lambda r: (r.r_score if r is not None else -1, r.num_requests_evaluated if r is not None else -1),
            reverse=True
        )
        # update ranks  
        for rank, report in enumerate(reports, start=1):
            report.rank = rank

        # --- Normalization step ---
        total_r_score = sum(r.r_score for r in reports if r.r_score > 0)
        for report in reports:
            if total_r_score > 0:
                this_r_score = report.r_score
                report.r_score = this_r_score / total_r_score                
            else:
                report.r_score = 0.0          
                print(f"⚠️ Total r_score is zero, cannot normalize scores.")
            # Update report_card with normalized score
            report.report_card = (
                f"Miner {report.miner_hotkey} scored: {report.r_score} with "
                f"{report.num_requests_evaluated} requests and {report.total_unique_products} products - "
                f"success rate: {report.num_success / (report.num_success + report.num_failures):.2%}"
            )
        # --- End normalization ---

        print(f"\033[32mScored {len(reports)} miners. Skipped {len(skipped_reports)} miners with < {MIN_REQUESTS} samples.\033[0m")
        return reports
    
    

    def score_miner(self, miner_hotkey: str, days_ago: int = 7, min_success: int = 50) -> MinerReport:
        """
            Get the last N responses of this miner identified by miner_hotkey
            and created_at > days_ago.
            For each response, extract the products and their reasons,
            score all reasons, and average per response.
            r_score is the average of per-response averages.
            f_score = r_score.
        """
        df = self.get_dataframe_by_miner(miner_hotkey)
        if df is None or df.empty:
            print(f"No data found for miner hotkey: {miner_hotkey}")
            return None
        df = df.copy()
        df['created_at'] = pd.to_datetime(df['created_at'], utc=True)
        
        cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=days_ago)
        filtered_df = df[
            (df['bt_header_axon_hotkey'] == miner_hotkey) &
            (df['created_at'] > cutoff)
        ]

        if filtered_df.empty:
            print(f"No data found for miner hotkey: {miner_hotkey}")
            return None        
        
        filtered_df = filtered_df.infer_objects(copy=False)
        #print(df.head())      

        report = MinerReport()
        report.evalator_notes = []
        created_at_val = filtered_df['created_at'].iloc[0] if not filtered_df['created_at'].empty else None
        report.created_at = created_at_val.strftime('%Y-%m-%d %H:%M:%S') if hasattr(created_at_val, 'strftime') else str(created_at_val) if created_at_val is not None else None

        report.miner_hotkey = str(miner_hotkey) if miner_hotkey is not None else ""
        #report.miner_uid = str(filtered_df['miner_uid'].iloc[0]) if not filtered_df['miner_uid'].empty else ""
        success_rows = filtered_df[filtered_df['bt_header_axon_status_code'] == '200']
        if not success_rows.empty and 'miner_uid' in success_rows.columns:
            #report.miner_uid = str(success_rows['miner_uid'].iloc[0])
            report.miner_uid = str(success_rows['miner_uid'].iloc[-1])  # Changed from iloc[0] to iloc[-1]
        else:
            report.miner_uid = ""
        report.miner_ip = str(filtered_df['bt_header_axon_ip'].iloc[0]) if not filtered_df['bt_header_axon_ip'].empty else ""
        report.num_success = int(len(filtered_df[filtered_df['bt_header_axon_status_code'] == '200']))
        report.num_failures = int(len(filtered_df[filtered_df['bt_header_axon_status_code'] != '200']))

        models_used = filtered_df['models_used'].dropna().unique()
        report.models_used = [m for m in models_used if isinstance(m, str) and m != "0.0"]    

        # Calculate average response time
        if 'bt_header_axon_process_time' in filtered_df.columns:
            avg_resp = pd.to_numeric(filtered_df['bt_header_axon_process_time'], errors='coerce').mean()
            report.avg_response_time = float(avg_resp) if avg_resp is not None else 0.0
            report.avg_response_time = round(report.avg_response_time, 4)
        else:
            report.avg_response_time = 0.0

        rows = len(filtered_df)
        print(f"\033[32mMiner {miner_hotkey} has {rows} responses in the last {days_ago} days. \033[0m")
    
        # For each response, score all products, average per response, then average those
        per_response_averages = []
        total_products = set()
        duplicate_penalty_count = 0  # Track how many responses had a significant duplicate penalty
        for idx, row in filtered_df.iterrows():
            result_val = row['results']
            if result_val:
                try:
                    reasoned_products = ReasonedProduct.from_json(result_val)
                    if reasoned_products:
                        scores = []
                        reasons = []
                        for product in reasoned_products:
                            total_products.add(product.sku)                            
                            score = self.score_reasoning_product(miner_hotkey, product)
                            scores.append(score)
                            reasons.append(product.reason)
                        if scores:
                            avg_score = sum(scores) / len(scores)
                            # Penalize for duplicate/near-duplicate reasons
                            penalty = self._duplicate_reason_penalty(reasons)
                            if penalty > 0.2:  # Consider 20%+ penalty as significant
                                duplicate_penalty_count += 1
                                print(f"⚠️ Duplicate/near-duplicate reasons detected for miner {miner_hotkey}, penalty: {penalty:.2f}")
                            avg_score = avg_score * (1 - penalty)                            
                            per_response_averages.append(avg_score)
                except Exception as e:
                    print(f"Error parsing results for miner {miner_hotkey}: {e}")

        print("\033[32m---------------SCORE MINER----------------- \033[0m")
        print(f"Miner {miner_hotkey} has {len(per_response_averages)}) responses with reasoned products.")
        # r_score: average of per-response averages
        if per_response_averages:
            miner_score = sum(per_response_averages) / len(per_response_averages)
        else:
            miner_score = 0.0
        print(f"Pre score for miner {miner_score:.2f} based on {len(per_response_averages)} responses.")
        # Check for duplicate reasons

        reason_dupe_threshold = 0.05
        if len(per_response_averages) > 0.0:
            frac_duplicate = duplicate_penalty_count / len(per_response_averages)
            if frac_duplicate > reason_dupe_threshold:
                print(f"❌ Miner {miner_hotkey} has {frac_duplicate} duplicate responses. Setting score to 0.")
                report.evalator_notes.append(f"{frac_duplicate} duplicate responses detected. Score set to 0.")
                miner_score = 0.0
        print(f"Score after duplicate penalty {miner_score:.2f}")

        #Check for excessive SKU duplication across queries
        sku_dupe_threshold = SKU_DUPLICATION_THRESHOLD
        if self.debug:
            sku_dupe_threshold = .60
        print(f"Checking for excessive SKU duplication for miner {miner_hotkey}")
        print(f"check_individual_sku_duplication row_count = {len(success_rows)}, threshold = {sku_dupe_threshold:.2%}")
        has_dupes, flagged_skus = self.check_individual_sku_duplication(success_rows, miner_hotkey, sku_dupe_threshold)     
        #if self.check_individual_sku_duplication(success_rows, threshold=sku_dupe_threshold):
        if has_dupes:        
            print(f"Miner {miner_hotkey} flagged for excessive SKU duplication across queries.")
            report.evalator_notes.append(f"Flagged for excessive SKU duplication across queries (> {sku_dupe_threshold:.0%} overlap).")
            if flagged_skus:
                report.evalator_notes.append(f"Flagged SKUs: {', '.join(flagged_skus)}")
            miner_score = 0.0
        print(f"Score after SKU duplication check {miner_score:.2f}")      

        # Check for cross site copying
        if self.check_cross_catalog_reasons(success_rows):
            miner_score = 0.0
            report.evalator_notes.append("Flagged for cross-catalog copying of reasons.")    
            print(f"Miner {miner_hotkey} flagged for cross-catalog copying of reasons.")            

       # Penalize for failures (non-200 status) - REDUCED EMPHASIS as consensus already handles this fairly
        success_count = int(len(filtered_df[filtered_df['bt_header_axon_status_code'] == '200']))
        total_count = len(filtered_df)
        success_rate = success_count / total_count if total_count > 0 else 0.0                
        penalty_multiplier = 1.0  # Start with no penalty       
        
        if total_count > 200:
            # Penalize for suspiciously perfect success rates (e.g., >98% indicates potential gaming)
            if success_rate > 0.98:
                over_perfect_penalty = 0.9  # Mild penalty (10% reduction) for near-perfect success
                penalty_multiplier *= over_perfect_penalty
                print(f"⚠️ Miner {miner_hotkey} has suspiciously high success rate ({success_rate:.2%}). Applying mild over-perfection penalty: {over_perfect_penalty:.2f} (new multiplier: {penalty_multiplier:.4f})")
                report.evalator_notes.append(f"Mild over-perfection penalty applied for {success_rate:.2%} success rate.")
            # Penalize for unrealistically fast response times
            if report.avg_response_time < 1.4 and report.avg_response_time > 0:
                speed_penalty = 0.65  # Reduce score by 35% for too-fast responses
                penalty_multiplier *= speed_penalty
                print(f"⚠️ Miner {miner_hotkey} response time ({report.avg_response_time:.2f}s) is suspiciously fast. Applying speed penalty: {speed_penalty:.2f} (new multiplier: {penalty_multiplier:.4f})")
                report.evalator_notes.append(f"Speed penalty applied for fast response time ({report.avg_response_time:.2f}s).")

            # if report.avg_response_time < 1.8 and report.avg_response_time > 0 and total_count > 400:
            #     speed_penalty = 0.80  # Reduce score by 20% for very fast responses or caching
            #     penalty_multiplier *= speed_penalty
            #     print(f"⚠️ Miner {miner_hotkey} response time ({report.avg_response_time:.2f}s) is very fast. Applying speed penalty: {speed_penalty:.2f} (new multiplier: {penalty_multiplier:.4f})")
            #     report.evalator_notes.append(f"Speed penalty applied for very fast response time ({report.avg_response_time:.2f}s).")

        # NEW: Bonus for realistic response times
        if 1.4 <= report.avg_response_time <= 8.0:
            realism_bonus = 1.15  # Boost score by 15% for realistic timing
            penalty_multiplier *= realism_bonus
            print(f"✅ Miner {miner_hotkey} has realistic response time ({report.avg_response_time:.2f}s). Applying realism bonus: {realism_bonus:.2f} (new multiplier: {penalty_multiplier:.4f})")
            report.evalator_notes.append(f"Realism bonus applied for response time ({report.avg_response_time:.2f}s).")

        # Apply the updated multiplier
        miner_score *= penalty_multiplier

        # Enforce minimum success count
        if len(filtered_df) < min_success:
            miner_score *= 0.05  # Apply 95% penalty instead of setting to 0.0
            report.evalator_notes.append(f"Only {len(filtered_df)} responses. Minimum is {min_success}. Applied 95% penalty.")
            #miner_score = 0.0

        if 1==1:
            if self.is_suspect_miner(miner_hotkey):
                miner_score = 0.0
                report.evalator_notes.append("Flagged as suspect miner for proof signature abuse")
                print(f"Miner {miner_hotkey} flagged as suspect miner for proof signature abuse")


        # Finalize report
        report.r_score = miner_score
        report.f_score = 0
        report.s_score = 0
        report.num_requests_evaluated = len(filtered_df)
        report.total_unique_products = len(total_products)
        print(f"Final Score for miner {miner_hotkey}:\033[31m {report.r_score:.2f} \033[0m")
        report.scored_at = datetime.now(tz=timezone.utc).strftime('%Y-%m-%d %H:%M:%S')
        return report

    
    def score_batch(self, batch_id: str, df: DataFrame) -> BatchReport:
        filtered_df = df[(df['site_key'] == batch_id)]
        if filtered_df.empty:
            print(f"No data found for batch ID: {batch_id}")
            return None
        report = BatchReport()
        report.keys = []
        report.scores = []
        report.axon_status_codes = []
        report.dendrite_status_codes = []
        report.batch_id = batch_id
        report.batch_date = filtered_df['created_at'].iloc[0] if not filtered_df['created_at'].empty else None
        report.query = filtered_df['query'].iloc[0] if not filtered_df['query'].empty else None
        #report.avg_process_time = filtered_df['bt_header_axon_process_time'].mean() if 'bt_header_axon_process_time' in filtered_df else None        
        report.avg_axon_process_time = pd.to_numeric(filtered_df['bt_header_axon_process_time'], errors='coerce').mean() if 'bt_header_axon_process_time' in filtered_df.columns else None
        report.avg_dendrite_process_time = pd.to_numeric(filtered_df['bt_header_dendrite_process_time'], errors='coerce').mean() if 'bt_header_dendrite_process_time' in filtered_df.columns else None
        report.num_results = filtered_df['num_results'].iloc[0] if not filtered_df['num_results'].empty else None
        try:
            report.batch_elected_miner_uid = filtered_df['batch_elected_uid'].mode().iloc[0] if not filtered_df['batch_elected_uid'].mode().empty else None
        except Exception as e:
            print(f"Error getting batch_elected_miner_uid: {e}")
            report.batch_elected_miner_uid = ""
        
        # Get the winning miner's model
        if report.batch_elected_miner_uid:
            winning_miner_row = filtered_df[filtered_df['miner_uid'] == report.batch_elected_miner_uid]
            if not winning_miner_row.empty:
                report.batch_elected_model = winning_miner_row['models_used'].iloc[0]
                report.batch_elected_result = winning_miner_row['results'].iloc[0] if 'results' in winning_miner_row.columns else None
                try:
                    report.batch_elected_miner_hotkey = winning_miner_row['bt_header_axon_hotkey'].iloc[0] if 'bt_header_axon_hotkey' in winning_miner_row.columns else None
                except Exception as e:
                    report.batch_elected_miner_hotkey = None

            else:
                report.batch_elected_model = "Unknown"
        else:
            report.batch_elected_model = "No Winner"        

        #print("----------------- STARTING BATCH REPORT -----------------")
        # print("\n=== DEBUG: Checking results by hotkey ===")
        # for hotkey in filtered_df["bt_header_axon_hotkey"].unique():
        #     hotkey_data = filtered_df[filtered_df["bt_header_axon_hotkey"] == hotkey]
        #     results_status = []
        #     for idx, row in hotkey_data.iterrows():
        #         result_val = row['results']
        #         if result_val is None:
        #             results_status.append("NULL")
        #         elif result_val == '':
        #             results_status.append("EMPTY")
        #         else:
        #             results_status.append(f"DATA({len(str(result_val))} chars)")
            
        #     print(f"{hotkey[:10]}...: {results_status}")
        # print("=== END DEBUG ===\n")    
        
        scores = []       
        for hotkey in filtered_df["bt_header_axon_hotkey"].unique():
            hotkey_data = filtered_df[filtered_df["bt_header_axon_hotkey"] == hotkey]            
            report.keys.append(hotkey)
            axon_code = hotkey_data['bt_header_axon_status_code'].mode().iloc[0] if not hotkey_data['bt_header_axon_status_code'].mode().empty else None
            report.axon_status_codes.append(axon_code)
            dendrite_code = hotkey_data['bt_header_dendrite_status_code'].mode().iloc[0] if not hotkey_data['bt_header_dendrite_status_code'].mode().empty else None
            report.dendrite_status_codes.append(dendrite_code)            
            
            hotkey_scores = []  # Collect scores for this hotkey
            for idx, row in hotkey_data.iterrows():
                result_val = row['results']
                if result_val is None:
                    hotkey_scores.append(0.0)
                elif result_val == '':
                    hotkey_scores.append(0.0)
                else:
                    try:
                        products = ReasonedProduct.from_json(result_val)
                        if products:
                            product_scores = []
                            for product in products:
                                score = self.score_reasoning_product(hotkey, product) 
                                product_scores.append(score)
                                print(f"Product '{product.name}' scored: {score} for reason: '{product.reason[:50]}...'")
                            
                            avg_score = sum(product_scores) / len(product_scores) if product_scores else 0.0
                            hotkey_scores.append(avg_score)
                        else:
                            print(f"No products parsed for hotkey {hotkey}")
                            hotkey_scores.append(0.0)
                    except Exception as e:
                        print(f"Error processing results for hotkey {hotkey}: {e}")
                        hotkey_scores.append(0.0)            
            
            final_hotkey_score = sum(hotkey_scores) / len(hotkey_scores) if hotkey_scores else 0.0
            scores.append(final_hotkey_score)
            print(f"Final score for hotkey {hotkey[:10]}...: {final_hotkey_score}")

        report.scores = scores

        
        print("----------------- END BATCH REPORT -----------------")
        return report
    

    # def check_sku_duplication(self, filtered_df, miner_hotkey, threshold=0.5):
    #     """
    #     Checks if a miner is returning the same SKUs for different queries.
    #     Flags if more than `threshold` fraction of queries have identical SKU sets.
    #     Prints the top 3 most common SKU sets with their counts and percentages.
    #     """
    #     from collections import Counter
    #     print(f"Checking SKU duplication for miner {miner_hotkey}...")
    #     print(f"Total queries status == 200 : {len(filtered_df)}")

    #     query_sku_map = {}
    #     for idx, row in filtered_df.iterrows():
    #         query = row.get('query')
    #         result_val = row.get('results')
    #         if query and result_val:
    #             try:
    #                 products = ReasonedProduct.from_json(result_val)
    #                 skus = tuple(sorted(p.sku for p in products if p.sku))
    #                 query_sku_map[query] = skus
    #             except Exception as e:
    #                 print(f"Error parsing results for miner {miner_hotkey}: {e}")

    #     total_queries = len(query_sku_map)
    #     sku_tuple_list = list(query_sku_map.values())
    #     counter = Counter(sku_tuple_list)
    #     most_common = counter.most_common(10)

    #     print(f"\n--- SKU Duplication Analysis for miner {miner_hotkey} ---")
    #     print(f"Total queries: {total_queries}")
    #     for i, (sku_set, count) in enumerate(most_common, 1):
    #         percent = (count / total_queries) * 100 if total_queries else 0
    #         print(f"Top {i}: {count} queries ({percent:.2f}%) returned SKUs: {sku_set}")

    #     if most_common:
    #         most_common_sku_set, most_common_count = most_common[0]
    #         if total_queries > 1 and (most_common_count / total_queries) > threshold:
    #             print(f"❌ Miner {miner_hotkey} is likely cheating: {most_common_count}/{total_queries} queries ({(most_common_count/total_queries)*100:.2f}%) returned the same SKUs: {most_common_sku_set}")
    #             return True
    #     return False

    def _duplicate_reason_penalty(self, reasons: list) -> float:
        if not reasons or len(reasons) == 1:
            return 0.0
        norm_reasons = [r.lower().strip() for r in reasons]
        unique = set(norm_reasons)
        if len(unique) == 1:
            return 1.0  # All are the same, total penalty
        total = len(norm_reasons)
        duplicates = 0
        for i in range(total):
            for j in range(i+1, total):
                if norm_reasons[i] == norm_reasons[j]:
                    duplicates += 1
        possible_pairs = total * (total - 1) / 2
        if possible_pairs == 0:
            return 0.0
        penalty = min(1.0, duplicates / possible_pairs)
        return penalty
    
        
    def check_individual_sku_duplication(self, filtered_df: pd.DataFrame, miner_hotkey:str, threshold:float=0.5) -> Tuple[bool, List[str]]:
        """
        Checks if any individual SKU appears in more than `threshold` fraction of queries.
        Prints the top 5 most common SKUs with their counts and percentages.
        Flags if any SKU exceeds the threshold.
        
        """
        from collections import Counter
        sku_counts = Counter()
        total_queries = 0
        seen_queries = set()
        for idx, row in filtered_df.iterrows():
            query = row.get('query')
            if not query or query in seen_queries:
                #print("SEEN QUERY, skipping...")
                continue
            result_val = row.get('results')
            if result_val:                
                try:
                    products = ReasonedProduct.from_json(result_val)
                    for p in products:
                        if p.sku:
                            sku_counts[p.sku] += 1
                    total_queries += 1
                    seen_queries.add(query)
                except Exception as e:
                    print(f"Error parsing results: {e}")

        print(f"\n--- Individual SKU Duplication Analysis for miner {miner_hotkey} ---")
        print(f"Total queries: {total_queries} - seen unique queries: {len(seen_queries)}")
        top_matches = sku_counts.most_common(5)
        for i, (sku, count) in enumerate(top_matches, 1):
            percent = (count / total_queries) * 100 if total_queries else 0
            print(f"Top {i}: SKU '{sku}' appeared in {count} queries ({percent:.2f}%)")

        flagged_skus = [sku for sku, count in sku_counts.items() if total_queries > 0 and (count / total_queries) > threshold]
        if flagged_skus:
            print(f"❌ SKUs appearing in >{threshold:.0%} of queries: {flagged_skus}")
            return True, flagged_skus
        return False, []
        

    def check_cross_catalog_reasons(self, filtered_df) -> bool:        
        terms = ["guitar", "drums", "instrument"]
        for idx, row in filtered_df.iterrows():
            result_val = row.get('results')
            query = row.get('query')
            woo_product = next((p for p in self.woo_catalog if p.sku.lower() == query.lower()), None)
            if woo_product and result_val:
                try:
                    products = ReasonedProduct.from_json(result_val)
                    for product in products:
                        if any(term in product.reason.lower() for term in terms):
                            print(f"⚠️ Cross-catalog copied reason detected for {woo_product.name} matching WooCommerce product {product.reason}")
                            return True
                except Exception as e:
                    print(f"Error parsing results: {e}")
                    continue

        return False

    
    def score_reasoning_product(self, hotkey: str, product: ReasonedProduct) -> float:
        """
        Scores a single ReasonedProduct's reasoning quality with comprehensive evaluation.
        Returns a float between 0 and 1.
        """
        # print(f"\n=== SCORING PRODUCT ===")
        # print(f"Hotkey: {hotkey[:10]}...")
        # print(f"Product: {product.name}")
        # print(f"Reason: '{product.reason}'")
        
        if not hotkey or not product or not product.reason:
            print("❌ Missing data - returning 0.0")
            return 0.0
        
        reason = product.reason.strip()
        if not reason:
            print("❌ Empty reason - returning 0.0")
            return 0.0
        
        # Non English
        if not reason.isascii():
            print(f"❌ Non-English characters detected in reason for hotkey {hotkey} - returning 0.0")
            return 0.0

        # Early exit for terrible reasons
        if self._is_terrible_reason(reason):
            print("❌ Terrible reason detected - returning 0.0")
            return 0.0
        
        # Early exit for too short reasons
        if len(reason) < 15:
            print(f"❌ Too short ({len(reason)} chars) - returning 0.05")
            return 0.05

        # Penalty if product name is in the reason (case-insensitive, punctuation removed)
        
        if product.name and product.reason:
            name = product.name.lower().strip()
            reason_lower = product.reason.lower().strip()
            name_clean = name.translate(str.maketrans('', '', string.punctuation))
            reason_clean = reason_lower.translate(str.maketrans('', '', string.punctuation))
            if name_clean in reason_clean or name in reason_lower:
                print(f"❌ Product name found in reason for hotkey {hotkey}, applying heavy penalty.")
                return 0.01  # or 0.0 for total disqualification

        #print(f"✅ Passed initial checks, reason length: {len(reason)}")
        
        # Initialize scoring components
        scores = {
            'content_quality': 0.0,
            'relevance': 0.0,
            'specificity': 0.0,
            'reasoning_depth': 0.0,
            'customer_focus': 0.0
        }
        
        # Weights for each component (should sum to 1.0)
        weights = {
            'content_quality': 0.25,
            'relevance': 0.35,
            'specificity': 0.20,
            'reasoning_depth': 0.15,
            'customer_focus': 0.05
        }
        
        try:
            scores['content_quality'] = self._evaluate_content_quality(reason, product)
            #print(f"Content quality: {scores['content_quality']}")
            
            scores['relevance'] = self._evaluate_relevance(reason, product)
            #print(f"Relevance: {scores['relevance']}")
            
            scores['specificity'] = self._evaluate_specificity(reason, product)
            #print(f"Specificity: {scores['specificity']}")
            
            scores['reasoning_depth'] = self._evaluate_reasoning_depth(reason, product)
            #print(f"Reasoning depth: {scores['reasoning_depth']}")
            
            scores['customer_focus'] = self._evaluate_customer_focus(reason, product)
            #print(f"Customer focus: {scores['customer_focus']}")
            
        except Exception as e:
            print(f"❌ Error in scoring components: {e}")
            return 0.0
        
        # Calculate weighted final score
        final_score = sum(scores[component] * weights[component] for component in scores)
        #print(f"Pre-penalty score: {final_score}")
        
        # Apply penalties
        final_score = self._apply_penalties(final_score, reason, product)
        #print(f"Post-penalty score: {final_score}")
        
        # Apply bonus for exceptional reasoning
        final_score = self._apply_excellence_bonus(final_score, reason, product)
        #print(f"Final score: {final_score}")

        return min(max(final_score, 0.0), 1.0)  # Ensure score is between 0.0 and 1.0        
        

    def _evaluate_content_quality(self, reason: str, product: ReasonedProduct) -> float:
        """Evaluates the basic quality of the reasoning content."""
        reason_lower = reason.lower().strip()
        
        # Already checked for terrible reasons in main method
        
        # More stringent length requirements
        if len(reason) < 20:
            return 0.2
        elif len(reason) < 40:
            return 0.4
        
        # Check for proper sentence structure
        has_punctuation = any(p in reason for p in '.!?')
        has_capitalization = reason[0].isupper() if reason else False
        has_multiple_sentences = len(re.split(r'[.!?]+', reason.strip())) >= 2
        
        # Check for coherent structure
        has_conjunctions = bool(re.search(r'\b(and|but|however|because|since|although|while|therefore)\b', reason_lower))
        
        base_score = 0.3
        if has_punctuation:
            base_score += 0.15
        if has_capitalization:
            base_score += 0.1
        if has_multiple_sentences:
            base_score += 0.2
        if has_conjunctions:
            base_score += 0.15
        if len(reason) > 80:
            base_score += 0.1
        
        return min(1.0, base_score)

    def _evaluate_relevance(self, reason: str, product: ReasonedProduct) -> float:
        """Evaluates how relevant the reasoning is to the specific product."""
        reason_words = set(re.findall(r'\b\w+\b', reason.lower()))
        product_name_words = set(re.findall(r'\b\w+\b', product.name.lower()))
        
        # Remove common stop words that don't add relevance
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        product_name_words = product_name_words - stop_words
        reason_words = reason_words - stop_words
        
        if not product_name_words:
            return 0.3  # Neutral score if no meaningful product words
        
        # Direct product name/feature mentions
        overlap = len(product_name_words.intersection(reason_words))
        name_relevance = min(overlap / len(product_name_words), 1.0)
        
        # Boost for exact product name mention
        product_name_in_reason = product.name.lower() in reason.lower()
        exact_name_bonus = 0.3 if product_name_in_reason else 0.0
        
        # SKU or specific model mentions (bonus)
        sku_bonus = 0.2 if product.sku and product.sku.lower() in reason.lower() else 0.0
        
        # Product-specific attribute mentions
        attribute_score = self._evaluate_product_attributes(reason, product)
        
        # Penalty for generic product references
        generic_penalty = self._calculate_generic_penalty(reason)
        
        relevance_score = (name_relevance * 0.4) + exact_name_bonus + sku_bonus + (attribute_score * 0.3)
        relevance_score = max(0.0, relevance_score - generic_penalty)
        
        return min(1.0, relevance_score)

    def _calculate_generic_penalty(self, reason: str) -> float:
        """Penalize for overly generic product references."""
        reason_lower = reason.lower()
        generic_terms = [
            'this product', 'this item', 'the product', 'the item',
            'it is', 'it has', 'it will', 'it can', 'it does'
        ]
        
        penalty = 0.0
        for term in generic_terms:
            if term in reason_lower:
                penalty += 0.05
        
        return min(0.3, penalty)  # Max 30% penalty

    def _evaluate_specificity(self, reason: str, product: ReasonedProduct) -> float:
        """Evaluates how specific and detailed the reasoning is."""
        reason_lower = reason.lower()
        
        # Look for specific details with higher standards
        specificity_indicators = [
            r'\b\d+\s*(inch|inches|cm|mm|ft|feet|lb|lbs|kg|oz|watts?|volts?|amps?)\b',  # Measurements with units
            r'\b(because|since|due to|thanks to|as a result|therefore)\b',  # Stronger causal language
            r'\b(enhance|improve|complement|pair|work with|suitable for|designed for|optimized for)\b',  # Functional language
            r'\b(professional|beginner|intermediate|advanced|expert|novice)\b',  # Skill level mentions
            r'\b(precision|accuracy|durability|reliability|performance|efficiency)\b',  # Quality metrics
            r'\b(comfortable|ergonomic|lightweight|heavy-duty|portable|compact)\b',  # Physical attributes
            r'\b(versatile|flexible|adjustable|customizable|modular)\b'  # Adaptability features
        ]
        
        specificity_count = sum(1 for pattern in specificity_indicators 
                               if re.search(pattern, reason_lower))
        
        # Enhanced use case patterns
        use_case_patterns = [
            r'\bfor (professional|amateur|home|studio|live|outdoor|indoor) (use|work|practice|performance)',
            r'\bwhen (you need|working with|performing|practicing|recording|creating)',
            r'\bin (professional|home|studio|live|commercial|educational) (settings?|environments?|applications?)',
            r'\bduring (performances?|practice|work|sessions?|recording|concerts?)'
        ]
        
        use_case_bonus = 0.25 if any(re.search(pattern, reason_lower) for pattern in use_case_patterns) else 0.0
        
        # Technical specification mentions
        tech_patterns = [
            r'\b(frequency|response|impedance|sensitivity|power|voltage|current)\b',
            r'\b(materials?|construction|build|design|engineering)\b',
            r'\b(features?|specifications?|capabilities?|functions?)\b'
        ]
        
        tech_bonus = 0.15 if any(re.search(pattern, reason_lower) for pattern in tech_patterns) else 0.0
        
        base_specificity = min(specificity_count / 5.0, 0.6)  # Harder to max out
        
        return min(1.0, base_specificity + use_case_bonus + tech_bonus)

    def _evaluate_product_attributes(self, reason: str, product: ReasonedProduct) -> float:
        """Evaluates mentions of general product attributes without category specificity."""
        reason_lower = reason.lower()
        
        # General quality and feature terms that apply to most products
        quality_terms = [
            'quality', 'durable', 'reliable', 'sturdy', 'lightweight', 'heavy-duty',
            'comfortable', 'ergonomic', 'user-friendly', 'easy to use', 'convenient',
            'efficient', 'effective', 'versatile', 'flexible', 'practical',
            'high-quality', 'well-made', 'well-built', 'solid', 'robust'
        ]
        
        # Functional terms
        functional_terms = [
            'works well', 'performs', 'functions', 'operates', 'delivers',
            'provides', 'offers', 'features', 'includes', 'comes with',
            'suitable for', 'perfect for', 'ideal for', 'great for'
        ]
        
        # Experience terms
        experience_terms = [
            'smooth', 'responsive', 'fast', 'quick', 'accurate', 'precise',
            'clear', 'bright', 'sharp', 'crisp', 'clean', 'stable'
        ]
        
        all_terms = quality_terms + functional_terms + experience_terms
        
        matches = sum(1 for term in all_terms if term in reason_lower)
        
        return min(1.0, matches / 3.0)  # Max score if 3+ relevant terms

    def _evaluate_reasoning_depth(self, reason: str, product: ReasonedProduct) -> float:
        """Evaluates the depth and sophistication of the reasoning."""
        reason_lower = reason.lower()
        
        # Look for sophisticated reasoning patterns
        depth_indicators = {
            'causal_reasoning': [
                r'\bbecause\b', r'\bsince\b', r'\bas\b', r'\bdue to\b',
                r'\btherefore\b', r'\bthus\b', r'\bhence\b'
            ],
            'comparative_reasoning': [
                r'\bbetter than\b', r'\bcompared to\b', r'\bversus\b',
                r'\balternative to\b', r'\binstead of\b'
            ],
            'conditional_reasoning': [
                r'\bif\b', r'\bunless\b', r'\bwhen\b', r'\bwhere\b',
                r'\bdepending on\b'
            ],
            'benefit_explanation': [
                r'\ballows\b', r'\benables\b', r'\bhelps\b', r'\bprovides\b',
                r'\boffers\b', r'\bdelivers\b'
            ]
        }
        
        depth_score = 0.0
        for category, patterns in depth_indicators.items():
            if any(re.search(pattern, reason_lower) for pattern in patterns):
                depth_score += 0.25
    
        # Bonus for multi-sentence reasoning
        sentence_count = len([s for s in re.split(r'[.!?]+', reason) if s.strip()])
        sentence_bonus = min(0.2, (sentence_count - 1) * 0.1)
    
        return min(1.0, depth_score + sentence_bonus)

    def _evaluate_customer_focus(self, reason: str, product: ReasonedProduct) -> float:
        """Evaluates how customer-focused the reasoning is."""
        reason_lower = reason.lower()
    
        # Look for customer-oriented language
        customer_indicators = [
            r'\byou\b', r'\byour\b', r'\bcustomer\b', r'\buser\b',
            r'\bperson\b', r'\banyone\b', r'\bsomeone\b'
        ]
        
        benefit_language = [
            r'\bwill (help|improve|enhance|make|allow)\b',
            r'\bcan (use|enjoy|benefit|experience)\b',
            r'\bmay (find|need|want|prefer)\b',
            r'\bgreat for\b', r'\bperfect for\b', r'\bideal for\b'
        ]
        
        customer_focus = sum(1 for pattern in customer_indicators 
                            if re.search(pattern, reason_lower))
        benefit_focus = sum(1 for pattern in benefit_language 
                           if re.search(pattern, reason_lower))
        
        return min(1.0, (customer_focus * 0.3) + (benefit_focus * 0.4))

    def _apply_excellence_bonus(self, score: float, reason: str, product: ReasonedProduct) -> float:
        """Apply bonus for exceptional reasoning quality."""
        reason_lower = reason.lower()
        
        excellence_indicators = [
            r'\b(however|nevertheless|furthermore|moreover|additionally|specifically)\b',  # Sophisticated connectors
            r'\b(ensures?|guarantees?|delivers?|provides?|enables?|facilitates?)\b',  # Strong benefit language
            r'\b(compared to|versus|alternative to|superior to|advantage over)\b',  # Comparative analysis
            r'\b(depending on|varies based on|tailored for|customized for)\b'  # Conditional reasoning
        ]
        
        excellence_count = sum(1 for pattern in excellence_indicators 
                              if re.search(pattern, reason_lower))
        
        # Bonus for detailed explanations (multiple sentences with good structure)
        sentences = [s.strip() for s in re.split(r'[.!?]+', reason) if s.strip()]
        detailed_bonus = 0.05 if len(sentences) >= 3 and all(len(s) > 10 for s in sentences) else 0.0
        
        excellence_bonus = min(0.15, excellence_count * 0.03) + detailed_bonus
        
        return min(1.0, score + excellence_bonus)

    def _apply_penalties(self, score: float, reason: str, product: ReasonedProduct) -> float:
        """Applies various penalties to reduce score for poor reasoning."""
        reason_lower = reason.lower()
        
        # Enhanced penalty for template language
        template_phrases = [
            "this product is", "this item is", "customers love", "users love",
            "highly rated", "top seller", "best choice", "popular choice",
            "don't miss out", "limited time", "special offer", "great deal",
            "must have", "can't go wrong", "you won't be disappointed"
        ]
        
        template_penalty = min(0.4, sum(0.08 for phrase in template_phrases if phrase in reason_lower))
        
        # Penalty for price-focused reasoning
        price_focused = len(re.findall(r'\b(cheap|expensive|price|cost|money|dollar|\$|budget|affordable)\b', reason_lower))
        price_penalty = min(0.25, price_focused * 0.08)
        
        # Penalty for overly promotional language
        promotional_words = ['amazing', 'incredible', 'fantastic', 'outstanding', 'exceptional', 'phenomenal', 'revolutionary']
        promo_penalty = min(0.2, sum(0.04 for word in promotional_words if word in reason_lower))
        
        # Penalty for repetitive language
        words = reason_lower.split()
        word_counts = {}
        for word in words:
            if len(word) > 3:  # Only check meaningful words
                word_counts[word] = word_counts.get(word, 0) + 1
        
        repetition_penalty = min(0.15, sum(0.02 for count in word_counts.values() if count > 2))
        
        total_penalty = template_penalty + price_penalty + promo_penalty + repetition_penalty
        
        return max(0.0, score - total_penalty)

    # def display_batch_report(self, report: BatchReport, theme: str = "default", save_png: bool = False, output_dir: str = ".") -> str:
    #     """
    #     Displays the batch report using the BatchReportPresenter.
    #     """
        
    #     presenter = BatchReportPresenter(theme=theme)
        
    #     # Create full dashboard
    #     saved_report = presenter.create_dashboard(report, save_png=save_png, output_dir=output_dir, show_plot=False, is_debug=self.debug)
        
    #     # Print console summary
    #     presenter.print_console_summary(report)

    #     return saved_report
    
    def _is_terrible_reason(self, reason: str) -> bool:
        """Check if reason is terrible using both exact matches and patterns."""
        reason_lower = reason.lower().strip()
        
        # Check exact matches
        if reason_lower in self.terrible_reasons:
            return True
        
        # Check for terrible patterns
        terrible_patterns = [
            r'^(very |really |so |pretty |quite )?(good|great|nice|cool|awesome)!*$',
            r'^(i |we )?(love|like|hate|dislike) (it|this)!*$',
            r'^(buy|get|purchase|avoid|skip) (this|it)!*$',
            r'^(yes|no|yep|nope|sure|maybe)!*$',
            r'^(wow|omg|lol|meh|ugh)!*$',
            r'^\d+/\d+$',  # Just ratings like "5/5" or "3/10"
            r'^\d+ stars?$',  # Just star ratings
            r'^[👍👎😍😊😢😡💯🔥]+$',  # Just emojis
            r'^.{1,5}$',  # Anything 5 characters or less
        ]
        
        return any(re.match(pattern, reason_lower) for pattern in terrible_patterns)
