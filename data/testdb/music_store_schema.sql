CREATE TABLE "music_products" (
    "sku" TEXT,
    "price" REAL,
    "name" TEXT
);

CREATE TABLE "music_order_items" (
    "order_id" INTEGER,
    "sku" TEXT,
    "price" REAL,
    "item_id" INTEGER,
    "name" TEXT,
    "qty" INTEGER,
    "row_total" REAL
);

CREATE TABLE "music_orders" (
    "order_id" TEXT,
    "grand_total" REAL,
    "status" TEXT,
    "subtotal" REAL,
    "subtotal_incl_tax" REAL,
    "subtotal_invoiced" REAL,
    "total_item_count" REAL,
    "total_paid" REAL,
    "total_qty_ordered" TEXT,
    "updated_at" TEXT,
    "group_id" TEXT
);

CREATE INDEX idx_orders_item_id ON music_order_items(order_id);
CREATE INDEX idx_sku_name ON music_products(sku, name);
CREATE INDEX idx_sku_name_items ON music_order_items(sku, name);
CREATE INDEX idx_orders_id ON music_orders(order_id);

CREATE VIEW v_orders_overview AS
SELECT 
    o.order_id,
    o.status,
    o.grand_total,
    o.subtotal,
    o.total_paid,
    o.total_qty_ordered,
    o.total_item_count,
    o.updated_at,
    o.group_id,
    ROUND(o.grand_total - o.subtotal, 2) AS estimated_tax_or_shipping,
    CASE 
        WHEN o.total_paid >= o.grand_total THEN 'Paid in full'
        WHEN o.total_paid > 0          THEN 'Partially paid'
        ELSE 'Not paid'
    END AS payment_status
FROM music_orders o;

CREATE VIEW v_orders_by_status AS
SELECT 
    status,
    COUNT(*)                  AS order_count,
    ROUND(SUM(grand_total), 2)   AS total_revenue,
    ROUND(SUM(total_paid), 2)    AS total_received,
    ROUND(SUM(subtotal), 2)      AS total_subtotal,
    SUM(total_item_count)        AS total_items_sold,
    ROUND(AVG(grand_total), 2)   AS avg_order_value,
    MIN(updated_at)              AS earliest_update,
    MAX(updated_at)              AS latest_update
FROM music_orders
GROUP BY status
ORDER BY order_count DESC;
CREATE VIEW v_music_orders_to_remove AS
select * From v_order_details
where item_name is null;
CREATE VIEW v_order_details AS
SELECT 
    o.order_id,
    o.status,
    o.grand_total,
    o.subtotal,
    o.total_paid,
    o.updated_at,
    o.total_item_count,
    
    i.item_id,
    i.sku,
    i.name          AS item_name,
    i.qty,
    i.price         AS item_price,
    i.row_total,
    
    p.name          AS product_name_from_catalog,
    p.price         AS catalog_price

FROM music_orders o
LEFT JOIN music_order_items i ON i.order_id = o.order_id
LEFT JOIN music_products   p ON p.sku     = i.sku;


UPDATE music_orders
SET updated_at = CASE
    WHEN LENGTH(updated_at) = 13 THEN
        -- Convert YY-MM-DD H:MM to YYYY-MM-DD HH:MM:SS
        '20' || SUBSTR(updated_at, 1, 2) || '-' ||  -- Year: 20YY
        SUBSTR(updated_at, 4, 2) || '-' ||          -- Month: MM
        SUBSTR(updated_at, 7, 2) || ' ' ||          -- Day: DD
        '0' || SUBSTR(updated_at, 10, 4) || ':00'   -- Time: 0H:MM + :00
    WHEN LENGTH(updated_at) = 14 THEN
        -- Convert YY-MM-DD HH:MM to YYYY-MM-DD HH:MM:SS
        '20' || SUBSTR(updated_at, 1, 2) || '-' ||  -- Year: 20YY
        SUBSTR(updated_at, 4, 2) || '-' ||          -- Month: MM
        SUBSTR(updated_at, 7, 2) || ' ' ||          -- Day: DD
        SUBSTR(updated_at, 10, 5) || ':00'          -- Time: HH:MM + :00
    ELSE
        -- Full format already good; leave as-is
        updated_at
END;