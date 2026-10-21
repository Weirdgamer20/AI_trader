from __future__ import annotations

import os
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Generator, Any

class DatabaseManager:
    """
    Unified Relational Database Manager.
    Supports PostgreSQL via connection string (DATABASE_URL) when configured,
    and defaults seamlessly to local SQLite with WAL mode, foreign keys,
    and identical relational schemas for deterministic Windows execution.
    """
    def __init__(self, db_path: str | Path | None = None, connection_url: str | None = None):
        self.connection_url = connection_url or os.environ.get("DATABASE_URL")
        self.is_postgres = bool(self.connection_url and self.connection_url.startswith(("postgres://", "postgresql://")))
        
        if not self.is_postgres:
            if db_path is None:
                default_dir = Path(__file__).resolve().parent.parent.parent / "data"
                default_dir.mkdir(parents=True, exist_ok=True)
                self.db_path = default_dir / "ledger.db"
            else:
                self.db_path = Path(db_path)
                self.db_path.parent.mkdir(parents=True, exist_ok=True)
        else:
            self.db_path = None

        self._init_schema()

    def get_connection(self) -> Any:
        if self.is_postgres:
            import psycopg2
            import psycopg2.extras
            conn = psycopg2.connect(self.connection_url)
            conn.autocommit = False
            return conn
        else:
            conn = sqlite3.connect(str(self.db_path), timeout=30.0)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA foreign_keys = ON;")
            conn.execute("PRAGMA journal_mode = WAL;")
            return conn

    @contextmanager
    def session(self) -> Generator[Any, None, None]:
        conn = self.get_connection()
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _init_schema(self) -> None:
        """Initialize relational schema for accounts, orders, fills, ledger, and tax."""
        schema_sql = """
        CREATE TABLE IF NOT EXISTS accounts (
            account_id TEXT PRIMARY KEY,
            account_type TEXT NOT NULL, -- 'PAPER' or 'LIVE'
            currency TEXT NOT NULL DEFAULT 'INR',
            starting_balance REAL NOT NULL DEFAULT 500.0,
            available_balance REAL NOT NULL DEFAULT 500.0,
            locked_balance REAL NOT NULL DEFAULT 0.0,
            target_balance REAL NOT NULL DEFAULT 100000.0,
            affordable_loss_ceiling REAL NOT NULL DEFAULT 250.0,
            cumulative_realized_loss REAL NOT NULL DEFAULT 0.0,
            is_active INTEGER NOT NULL DEFAULT 1,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS orders (
            order_id TEXT PRIMARY KEY,
            client_order_id TEXT UNIQUE NOT NULL,
            account_id TEXT NOT NULL,
            market_pair TEXT NOT NULL,
            side TEXT NOT NULL, -- 'BUY' or 'SELL'
            order_type TEXT NOT NULL, -- 'LIMIT', 'MARKET'
            status TEXT NOT NULL, -- 'PROPOSED', 'RISK_CHECKED', 'AUTHORIZED', 'SUBMITTED', 'FILLED', 'PARTIAL', 'CANCELLED', 'REJECTED'
            price REAL,
            quantity REAL NOT NULL,
            filled_quantity REAL NOT NULL DEFAULT 0.0,
            planned_risk_inr REAL NOT NULL DEFAULT 0.0,
            stop_loss_price REAL,
            take_profit_price REAL,
            rejection_reason TEXT,
            idempotency_key TEXT UNIQUE,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            FOREIGN KEY (account_id) REFERENCES accounts(account_id)
        );

        CREATE TABLE IF NOT EXISTS fills (
            fill_id TEXT PRIMARY KEY,
            order_id TEXT NOT NULL,
            exchange_trade_id TEXT,
            market_pair TEXT NOT NULL,
            side TEXT NOT NULL,
            price REAL NOT NULL,
            quantity REAL NOT NULL,
            fee_inr REAL NOT NULL DEFAULT 0.0,
            gst_on_fee_inr REAL NOT NULL DEFAULT 0.0,
            tds_deducted_inr REAL NOT NULL DEFAULT 0.0,
            net_proceeds_inr REAL NOT NULL,
            timestamp TEXT NOT NULL,
            is_paper INTEGER NOT NULL DEFAULT 1,
            FOREIGN KEY (order_id) REFERENCES orders(order_id)
        );

        CREATE TABLE IF NOT EXISTS positions (
            position_id TEXT PRIMARY KEY,
            account_id TEXT NOT NULL,
            market_pair TEXT NOT NULL,
            quantity REAL NOT NULL DEFAULT 0.0,
            average_entry_price REAL NOT NULL DEFAULT 0.0,
            cost_basis_inr REAL NOT NULL DEFAULT 0.0,
            current_price REAL NOT NULL DEFAULT 0.0,
            unrealized_pnl_inr REAL NOT NULL DEFAULT 0.0,
            realized_pnl_inr REAL NOT NULL DEFAULT 0.0,
            is_open INTEGER NOT NULL DEFAULT 1,
            updated_at TEXT NOT NULL,
            FOREIGN KEY (account_id) REFERENCES accounts(account_id)
        );

        CREATE TABLE IF NOT EXISTS ledger_entries (
            entry_id TEXT PRIMARY KEY,
            account_id TEXT NOT NULL,
            entry_type TEXT NOT NULL, -- 'DEPOSIT', 'BUY_COST', 'SELL_PROCEEDS', 'FEE', 'GST', 'TDS', 'PNL_ADJUSTMENT'
            amount_inr REAL NOT NULL,
            balance_after_inr REAL NOT NULL,
            reference_id TEXT, -- e.g. fill_id or order_id
            description TEXT,
            timestamp TEXT NOT NULL,
            FOREIGN KEY (account_id) REFERENCES accounts(account_id)
        );

        CREATE TABLE IF NOT EXISTS tax_records (
            record_id TEXT PRIMARY KEY,
            account_id TEXT NOT NULL,
            financial_year TEXT NOT NULL,
            fill_id TEXT NOT NULL,
            market_pair TEXT NOT NULL,
            sale_proceeds_inr REAL NOT NULL,
            cost_basis_inr REAL NOT NULL,
            economic_gain_inr REAL NOT NULL,
            taxable_gain_vda_inr REAL NOT NULL, -- Section 115BBH (No loss set-off)
            tax_liability_30pct_inr REAL NOT NULL,
            tds_withheld_1pct_inr REAL NOT NULL, -- Section 194S
            timestamp TEXT NOT NULL,
            FOREIGN KEY (account_id) REFERENCES accounts(account_id),
            FOREIGN KEY (fill_id) REFERENCES fills(fill_id)
        );

        CREATE TABLE IF NOT EXISTS audit_events (
            event_id TEXT PRIMARY KEY,
            event_type TEXT NOT NULL, -- 'RISK_HALT', 'EMERGENCY_STOP', 'ORDER_TRANSITION', 'MODEL_PROMOTION', 'SECRET_ACCESS', 'RECONCILIATION'
            severity TEXT NOT NULL, -- 'INFO', 'WARNING', 'CRITICAL', 'EMERGENCY'
            message TEXT NOT NULL,
            details_json TEXT,
            timestamp TEXT NOT NULL
        );
        """
        with self.session() as conn:
            cur = conn.cursor()
            for statement in schema_sql.split(";"):
                stmt = statement.strip()
                if stmt:
                    cur.execute(stmt)
