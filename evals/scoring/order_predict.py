import sqlite3
from dotenv import load_dotenv
load_dotenv()


class OrderForecasting:
    def __init__(self, db_connection):
        self.db = db_connection
    
    def _validate_inputs(self, sku: str, rec_skus: list[str]) -> tuple[str, list[str]]:
        """Validate and clean inputs"""
        if not sku or not isinstance(sku, str):
            raise ValueError("SKU must be a non-empty string")        
        if not rec_skus or not isinstance(rec_skus, list):
            raise ValueError("rec_skus must be a non-empty list")
        cleaned_rec_skus = []
        for rec_sku in rec_skus:
            if rec_sku and isinstance(rec_sku, str):
                cleaned = rec_sku.strip()
                if cleaned and cleaned != sku:  # Don't include the same SKU
                    cleaned_rec_skus.append(cleaned)
        
        if not cleaned_rec_skus:
            raise ValueError("No valid recommendation SKUs provided")
            
        return sku.strip(), cleaned_rec_skus
    
    def find_similar_orders(self, sku: str, rec_skus: list[str]) -> dict:
        """Find orders that contain both the given SKU and one or more recommendation SKUs."""
        try:
            sku, rec_skus = self._validate_inputs(sku, rec_skus)
        except ValueError as e:
            return {'orders': [], 'co_occurrence_stats': {}, 'total_orders': 0, 'error': str(e)}
        
        rec_sku_placeholders = ','.join(['?' for _ in rec_skus])        
        query = f"""
        SELECT DISTINCT o.order_id,
               o.grand_total,
               o.status,
               o.subtotal,
               o.total_item_count,
               o.total_paid,
               o.total_qty_ordered,
               o.updated_at,
               o.group_id,
               GROUP_CONCAT(DISTINCT oi_rec.sku) as matching_rec_skus,
               GROUP_CONCAT(DISTINCT oi_rec.name) as matching_rec_names,
               COUNT(DISTINCT oi_rec.sku) as rec_sku_count,
               SUM(oi_rec.row_total) as rec_items_total
        FROM music_orders o
        JOIN music_order_items oi_main ON o.order_id = oi_main.order_id
        JOIN music_order_items oi_rec ON o.order_id = oi_rec.order_id
        WHERE oi_main.sku = ?
        AND oi_rec.sku IN ({rec_sku_placeholders})
        AND oi_main.sku != oi_rec.sku        
        GROUP BY o.order_id
        ORDER BY rec_sku_count DESC, o.grand_total DESC
        """        
        
        try:
            params = [sku] + rec_skus
            cursor = self.db.execute(query, params)
            orders = cursor.fetchall()        
            co_occurrence_stats = self._get_co_occurrence_stats(sku, rec_skus)        
            return {
                'orders': [dict(order) for order in orders],
                'co_occurrence_stats': co_occurrence_stats,
                'total_orders': len(orders)
            }
        except sqlite3.Error as e:
            return {'orders': [], 'co_occurrence_stats': {}, 'total_orders': 0, 'error': f"Database error: {e}"}
    
    def _get_co_occurrence_stats(self, sku: str, rec_skus: list[str]) -> dict:
        """Get count of orders for each recommendation SKU with the given SKU"""
        stats = {}
        
        for rec_sku in rec_skus:
            query = """
            SELECT COUNT(DISTINCT o.order_id) as order_count
            FROM music_orders o
            WHERE o.order_id IN (
                SELECT order_id FROM music_order_items WHERE sku = ?
            )
            AND o.order_id IN (
                SELECT order_id FROM music_order_items WHERE sku = ?
            )
            """
            cursor = self.db.execute(query, [sku, rec_sku])
            result = cursor.fetchone()
            stats[rec_sku] = result[0] if result else 0 
            
        return stats
    
    def get_order_details_with_items(self, sku: str, rec_skus: list[str]) -> list:       
        if not sku or not rec_skus:
            return []
        
        rec_sku_placeholders = ','.join(['?' for _ in rec_skus])        
        query = f"""
        SELECT 
            o.order_id,
            o.grand_total,
            o.status,
            o.updated_at,
            oi.sku,
            oi.name,
            oi.price,
            oi.qty,
            oi.row_total,
            CASE 
                WHEN oi.sku = ? THEN 'target_sku'
                WHEN oi.sku IN ({rec_sku_placeholders}) THEN 'recommendation_sku'
                ELSE 'other_item'
            END as item_type
        FROM music_orders o
        JOIN music_order_items oi ON o.order_id = oi.order_id
        WHERE o.order_id IN (
            SELECT DISTINCT o2.order_id
            FROM music_orders o2
            JOIN music_order_items oi_main ON o2.order_id = oi_main.order_id
            JOIN music_order_items oi_rec ON o2.order_id = oi_rec.order_id
            WHERE oi_main.sku = ?
            AND oi_rec.sku IN ({rec_sku_placeholders})
            AND oi_main.sku != oi_rec.sku
        )
        ORDER BY o.order_id, oi.item_id
        """
        
        # Parameters: sku for CASE, rec_skus for CASE, sku for WHERE, rec_skus for WHERE
        params = [sku] + rec_skus + [sku] + rec_skus
        cursor = self.db.execute(query, params)
        return [dict(row) for row in cursor.fetchall()]
    
    def get_recommendation_strength(self, sku: str, rec_skus: list[str]) -> dict:
        """
        Calculate recommendation strength based on co-occurrence frequency.
        """
        # Get total orders containing the main SKU
        total_sku_orders_query = """
        SELECT COUNT(DISTINCT order_id) as total_orders
        FROM music_order_items 
        WHERE sku = ?
        """
        cursor = self.db.execute(total_sku_orders_query, [sku])
        result = cursor.fetchone()
        total_sku_orders = result[0] if result else 0
        
        if total_sku_orders == 0:
            return {'recommendations': [], 'base_sku_orders': 0}
        
        # Use existing co-occurrence method
        co_occurrence_stats = self._get_co_occurrence_stats(sku, rec_skus)
        
        # Get additional details for each recommendation
        recommendations = []
        for rec_sku, co_count in co_occurrence_stats.items():
            if co_count > 0:
                strength = (co_count / total_sku_orders) * 100
                
                # Get revenue details for this rec_sku in co-occurring orders
                revenue_query = """
                SELECT 
                    AVG(o.grand_total) as avg_order_value,
                    SUM(oi.row_total) as total_rec_revenue,
                    AVG(oi.price) as avg_rec_price,
                    COUNT(oi.item_id) as total_rec_items_sold
                FROM music_order_items oi
                JOIN music_orders o ON oi.order_id = o.order_id
                WHERE oi.sku = ?
                AND oi.order_id IN (
                    SELECT order_id FROM music_order_items WHERE sku = ?
                )
                """
                cursor = self.db.execute(revenue_query, [rec_sku, sku])
                revenue_result = cursor.fetchone()
                
                recommendations.append({
                    'sku': rec_sku,
                    'co_occurrence_count': co_count,
                    'strength_percentage': round(strength, 2),
                    'confidence': 'high' if strength > 10 else 'medium' if strength > 5 else 'low',
                    'avg_order_value': round(revenue_result[0], 2) if revenue_result and revenue_result[0] else 0,
                    'total_revenue': round(revenue_result[1], 2) if revenue_result and revenue_result[1] else 0,
                    'avg_item_price': round(revenue_result[2], 2) if revenue_result and revenue_result[2] else 0,
                    'total_items_sold': revenue_result[3] if revenue_result and revenue_result[3] else 0
                })
        
        recommendations.sort(key=lambda x: x['strength_percentage'], reverse=True)        
        return {
            'recommendations': recommendations,
            'base_sku_orders': total_sku_orders
        }
    
    def get_customer_purchase_patterns(self, sku: str, rec_skus: list[str]) -> dict:
        """
        Analyze purchase patterns using group_id as customer identifier
        """
        if not sku or not rec_skus:
            return {'patterns': [], 'total_customers': 0}
        
        rec_sku_placeholders = ','.join(['?' for _ in rec_skus])
        
        # FIXED: Use subqueries to find orders with both SKUs
        query = f"""
        SELECT 
            o.group_id as customer_id,
            COUNT(DISTINCT o.order_id) as total_orders,
            MIN(o.updated_at) as first_purchase_date,
            MAX(o.updated_at) as last_purchase_date,
            SUM(o.grand_total) as total_spent,
            GROUP_CONCAT(DISTINCT rec_items.sku) as purchased_rec_skus,
            COUNT(DISTINCT rec_items.sku) as unique_rec_skus_bought
        FROM music_orders o
        JOIN music_order_items main_items ON o.order_id = main_items.order_id
        JOIN music_order_items rec_items ON o.order_id = rec_items.order_id
        WHERE main_items.sku = ?
        AND rec_items.sku IN ({rec_sku_placeholders})
        AND main_items.sku != rec_items.sku
        AND o.group_id IS NOT NULL
        --AND o.status = 'complete'
        GROUP BY o.group_id
        ORDER BY unique_rec_skus_bought DESC, total_spent DESC
        """
    
        params = [sku] + rec_skus
        cursor = self.db.execute(query, params)
        patterns = cursor.fetchall()
        
        return {
            'patterns': [dict(pattern) for pattern in patterns],
            'total_customers': len(patterns)
        }

    def find_sequential_orders(self, sku: str, rec_skus: list[str]) -> dict:
        """Find orders where customers bought the SKU first, then later bought rec_skus."""
        try:
            sku, rec_skus = self._validate_inputs(sku, rec_skus)
        except ValueError as e:
            return self._empty_sequential_result(error=str(e))
        
        rec_sku_placeholders = ','.join(['?' for _ in rec_skus])
        
        query = f"""
        SELECT 
            first_order.group_id as customer_id,
            first_order.order_id as first_order_id,
            first_order.updated_at as first_order_date,
            first_order.grand_total as first_order_total,
            second_order.order_id as second_order_id,
            second_order.updated_at as second_order_date,
            second_order.grand_total as second_order_total,
            oi_rec.sku as purchased_rec_sku,
            oi_rec.name as purchased_rec_name,
            oi_rec.price as purchased_rec_price,
            oi_rec.qty as purchased_rec_qty,
            oi_rec.row_total as purchased_rec_total,
            JULIANDAY(second_order.updated_at) - JULIANDAY(first_order.updated_at) as days_between
        FROM music_orders first_order
        JOIN music_order_items oi_first ON first_order.order_id = oi_first.order_id
        JOIN music_orders second_order ON first_order.group_id = second_order.group_id
        JOIN music_order_items oi_rec ON second_order.order_id = oi_rec.order_id
        WHERE oi_first.sku = ?
        AND oi_rec.sku IN ({rec_sku_placeholders})
        AND first_order.updated_at < second_order.updated_at
        AND first_order.group_id IS NOT NULL        
        AND first_order.order_id != second_order.order_id
        ORDER BY first_order.group_id, first_order.updated_at, second_order.updated_at
        """
        
        try:
            params = [sku] + rec_skus
            cursor = self.db.execute(query, params)
            results = cursor.fetchall()            
            if not results:
                return self._empty_sequential_result()
            
            return self._process_sequential_results(results)
            
        except sqlite3.Error as e:
            return self._empty_sequential_result(error=f"Database error: {e}")
    
    def _empty_sequential_result(self, error=None):
        """Return empty sequential result structure"""
        result = {
            'sequential_patterns': [], 
            'customers': [], 
            'summary_stats': {
                'total_customers': 0,
                'total_sequential_orders': 0,
                'avg_days_between_purchases': 0,
                'total_rec_revenue': 0,
                'rec_sku_frequency': {},
                'rec_sku_unique_customers': {},
                'conversion_rate_by_sku': {},
                'avg_purchases_per_customer_by_sku': {}
            }
        }
        if error:
            result['error'] = error
        return result
    
    def _process_sequential_results(self, results):
        """Process sequential query results into structured data"""
        sequential_patterns = [dict(row) for row in results]
        
        # Group by customer for customer-level analysis
        customers = {}
        for pattern in sequential_patterns:
            customer_id = pattern['customer_id']
            if customer_id not in customers:
                customers[customer_id] = {
                    'customer_id': customer_id,
                    'first_purchase_date': pattern['first_order_date'],
                    'first_order_id': pattern['first_order_id'],
                    'first_order_total': pattern['first_order_total'],
                    'subsequent_purchases': [],
                    'total_rec_spending': 0,
                    'unique_rec_skus': set(),
                    'avg_days_between_purchases': 0
                }
            
            customers[customer_id]['subsequent_purchases'].append({
                'order_id': pattern['second_order_id'],
                'order_date': pattern['second_order_date'],
                'order_total': pattern['second_order_total'],
                'sku': pattern['purchased_rec_sku'],
                'name': pattern['purchased_rec_name'],
                'price': pattern['purchased_rec_price'],
                'qty': pattern['purchased_rec_qty'],
                'item_total': pattern['purchased_rec_total'] or 0,  # FIXED: Handle None
                'days_after_first': pattern['days_between']
            })
            
            customers[customer_id]['total_rec_spending'] += pattern['purchased_rec_total'] or 0
            customers[customer_id]['unique_rec_skus'].add(pattern['purchased_rec_sku'])
        
        # Calculate summary statistics with better error handling
        customer_list = list(customers.values())
        for customer in customer_list:
            customer['unique_rec_skus'] = list(customer['unique_rec_skus'])
            if customer['subsequent_purchases']:
                valid_days = [
                    p['days_after_first'] for p in customer['subsequent_purchases'] 
                    if p['days_after_first'] is not None and p['days_after_first'] >= 0
                ]
                customer['avg_days_between_purchases'] = (
                    sum(valid_days) / len(valid_days) if valid_days else 0
                )

        # Overall summary stats
        total_customers = len(customer_list)
        total_sequential_orders = len(sequential_patterns)

        valid_overall_days = [
            p['days_between'] for p in sequential_patterns 
            if p['days_between'] is not None and p['days_between'] >= 0
        ]
        avg_days_between = sum(valid_overall_days) / len(valid_overall_days) if valid_overall_days else 0
        total_rec_revenue = sum(c['total_rec_spending'] for c in customer_list)
        
        # Count frequency and unique customers per SKU
        rec_sku_frequency = {}
        rec_sku_unique_customers = {}
        
        for pattern in sequential_patterns:
            sku_bought = pattern['purchased_rec_sku']
            customer_id = pattern['customer_id']
            
            rec_sku_frequency[sku_bought] = rec_sku_frequency.get(sku_bought, 0) + 1
            
            if sku_bought not in rec_sku_unique_customers:
                rec_sku_unique_customers[sku_bought] = set()
            rec_sku_unique_customers[sku_bought].add(customer_id)
        
        return {
            'sequential_patterns': sequential_patterns,
            'customers': customer_list,
            'summary_stats': {
                'total_customers': total_customers,
                'total_sequential_orders': total_sequential_orders,
                'avg_days_between_purchases': round(avg_days_between, 1),
                'total_rec_revenue': round(total_rec_revenue, 2),
                'rec_sku_frequency': rec_sku_frequency,
                'rec_sku_unique_customers': {
                    sku: len(customers_set) 
                    for sku, customers_set in rec_sku_unique_customers.items()
                },
                'conversion_rate_by_sku': {
                    sku: round((len(rec_sku_unique_customers[sku]) / total_customers) * 100, 2) 
                    if total_customers > 0 else 0
                    for sku in rec_sku_frequency.keys()
                },
                'avg_purchases_per_customer_by_sku': {
                    sku: round(rec_sku_frequency[sku] / len(rec_sku_unique_customers[sku]), 2)
                    for sku in rec_sku_frequency.keys()
                }
            }
        }