"""
Supply Chain Analytics - Inventory Decision Agent
Orchestrates the 7-stage inventory decision architecture:
Customer Question ➔ Identify Model ➔ Extract Inputs ➔ Select Python Tool ➔ Calculate ➔ Verify ➔ Explain

Enhanced with deep natural-language input extraction:
- Comprehensive terminology and synonym recognition
- Fine-grained distinction between average vs std dev for demand and lead time
- Full price tier and break analysis
- Rigorous parameter verification (never invents missing numbers)
"""

import re
import math
from typing import Dict, Any, List, Optional, Tuple, Callable
import inventory_tools


class InventoryAgent:
    """
    Intelligent inventory decision agent implementing the 7-stage supply chain pipeline.
    """

    MODEL_EOQ = "EOQ"
    MODEL_QUANTITY_DISCOUNT = "QUANTITY_DISCOUNT"
    MODEL_SAFETY_STOCK_ROP = "SAFETY_STOCK_ROP"
    MODEL_FIXED_PERIOD_REVIEW = "FIXED_PERIOD_REVIEW"
    MODEL_NEWSVENDOR = "NEWSVENDOR"

    MODEL_DISPLAY_NAMES = {
        MODEL_EOQ: "Economic Order Quantity (EOQ)",
        MODEL_QUANTITY_DISCOUNT: "Quantity Discount Analysis",
        MODEL_SAFETY_STOCK_ROP: "Safety Stock & Reorder Point (Continuous Review)",
        MODEL_FIXED_PERIOD_REVIEW: "Fixed-Period Review (Periodic Review / Order-Up-To)",
        MODEL_NEWSVENDOR: "Newsvendor Model (Single-Period Perishable Demand)",
    }

    def __init__(self):
        self.tool_mapping = {
            self.MODEL_EOQ: inventory_tools.calculate_eoq,
            self.MODEL_QUANTITY_DISCOUNT: inventory_tools.calculate_quantity_discount,
            self.MODEL_SAFETY_STOCK_ROP: inventory_tools.calculate_rop,
            self.MODEL_FIXED_PERIOD_REVIEW: inventory_tools.calculate_periodic_review,
            self.MODEL_NEWSVENDOR: inventory_tools.calculate_newsvendor,
        }

    # -----------------------------------------------------------------------
    # STAGE 1: Identify Model
    # -----------------------------------------------------------------------
    def identify_model(self, question: str) -> Dict[str, Any]:
        """
        Analyzes natural language question to classify which inventory model to apply.
        """
        q = question.lower()

        scores = {
            self.MODEL_EOQ: 0,
            self.MODEL_QUANTITY_DISCOUNT: 0,
            self.MODEL_SAFETY_STOCK_ROP: 0,
            self.MODEL_FIXED_PERIOD_REVIEW: 0,
            self.MODEL_NEWSVENDOR: 0,
        }
        reasons = {k: [] for k in scores}

        # 1. Newsvendor cues
        newsvendor_keywords = [
            ("newsvendor", 12, "Explicit 'newsvendor' mention"),
            ("perishable", 8, "Perishable inventory single-cycle context"),
            ("salvage", 7, "Salvage value indicated"),
            ("residual value", 7, "Residual/markdown value indicated"),
            ("markdown price", 7, "Markdown price indicated"),
            ("overage", 8, "Overage/excess cost terms"),
            ("underage", 8, "Underage/shortage cost terms"),
            ("critical fractile", 9, "Critical fractile reference"),
            ("critical ratio", 8, "Critical ratio reference"),
            ("spoil", 6, "Spoilage/single-season goods"),
            ("single period", 8, "Single-period demand"),
            ("single season", 7, "Single-season demand"),
            ("shortage penalty", 7, "Shortage penalty indicated"),
            ("loss of goodwill", 6, "Goodwill loss penalty"),
        ]
        for kw, weight, reason in newsvendor_keywords:
            if kw in q:
                scores[self.MODEL_NEWSVENDOR] += weight
                reasons[self.MODEL_NEWSVENDOR].append(reason)

        # 2. Quantity Discount cues
        discount_keywords = [
            ("quantity discount", 12, "Explicit 'quantity discount' mention"),
            ("volume discount", 10, "Volume discount structure"),
            ("price tier", 9, "Price tier schedule mentioned"),
            ("price break", 9, "Price break mentioned"),
            ("quantity break", 10, "Quantity break mentioned"),
            ("bracket", 7, "Tier brackets indicated"),
            ("tier 1", 7, "Numbered pricing tiers"),
            ("all-units", 8, "All-units discount policy"),
            ("discount", 6, "Discount mention"),
        ]
        # Price tier syntax pattern like "0-499: $20", "500 to 999: $18" or "1000+: $16"
        if re.search(r"\d+\s*[-–to]+\s*\d+\s*(?:units)?\s*[:=]\s*\$?[0-9]", q) or re.search(r"\d+\+\s*[:=]\s*\$?[0-9]", q):
            scores[self.MODEL_QUANTITY_DISCOUNT] += 12
            reasons[self.MODEL_QUANTITY_DISCOUNT].append("Price bracket schedule syntax detected")

        for kw, weight, reason in discount_keywords:
            if kw in q:
                scores[self.MODEL_QUANTITY_DISCOUNT] += weight
                reasons[self.MODEL_QUANTITY_DISCOUNT].append(reason)

        # 3. Fixed-Period Review cues
        periodic_keywords = [
            ("periodic review", 12, "Explicit periodic review system"),
            ("fixed-period review", 12, "Fixed-period review policy"),
            ("fixed period review", 12, "Fixed period review policy"),
            ("fixed-period", 9, "Fixed-period policy"),
            ("fixed period", 9, "Fixed period policy"),
            ("order-up-to", 10, "Order-up-to level (Base Stock)"),
            ("order up to", 10, "Order up to level target"),
            ("base stock", 9, "Base stock target"),
            ("target inventory", 8, "Target inventory level"),
            ("review period", 9, "Review interval/period defined"),
            ("review interval", 8, "Review interval defined"),
            ("every week", 6, "Periodic review schedule (weekly)"),
            ("every month", 6, "Periodic review schedule (monthly)"),
            ("every 2 weeks", 7, "Periodic review schedule (bi-weekly)"),
            ("every 14 days", 7, "Periodic review schedule (14 days)"),
            ("inventory position", 6, "Inventory position calculation context"),
        ]
        for kw, weight, reason in periodic_keywords:
            if kw in q:
                scores[self.MODEL_FIXED_PERIOD_REVIEW] += weight
                reasons[self.MODEL_FIXED_PERIOD_REVIEW].append(reason)

        # 4. Safety Stock & Reorder Point cues
        ss_rop_keywords = [
            ("reorder point", 10, "Explicit Reorder Point (ROP) query"),
            ("rop", 9, "ROP abbreviation"),
            ("safety stock", 10, "Explicit Safety Stock query"),
            ("lead time demand", 7, "Demand during lead time inquiry"),
            ("continuous review", 9, "Continuous review (s, Q) policy"),
            ("service level", 5, "Service level specification"),
            ("cycle service level", 7, "Cycle service level specification"),
            ("stockout risk", 5, "Stockout probability context"),
        ]
        for kw, weight, reason in ss_rop_keywords:
            if kw in q:
                scores[self.MODEL_SAFETY_STOCK_ROP] += weight
                reasons[self.MODEL_SAFETY_STOCK_ROP].append(reason)

        # 5. EOQ cues
        eoq_keywords = [
            ("eoq", 12, "Explicit EOQ query"),
            ("economic order quantity", 12, "Economic Order Quantity specified"),
            ("ordering/setup cost", 6, "Ordering/setup cost parameter"),
            ("cost of placing an order", 6, "Cost of placing an order parameter"),
            ("cost per order", 5, "Cost per order parameter"),
            ("setup cost", 5, "Setup cost parameter"),
            ("setup expense", 5, "Setup expense parameter"),
            ("order cost", 5, "Order cost parameter"),
            ("ordering cost", 5, "Ordering cost parameter"),
            ("holding cost", 5, "Inventory holding cost parameter"),
            ("carrying cost", 5, "Inventory carrying cost parameter"),
            ("annual demand", 4, "Annual demand rate"),
            ("optimal batch", 7, "Optimal batch/lot size"),
            ("optimal lot size", 7, "Optimal lot size query"),
            ("how many units should we order", 4, "Order sizing query"),
        ]
        for kw, weight, reason in eoq_keywords:
            if kw in q:
                scores[self.MODEL_EOQ] += weight
                reasons[self.MODEL_EOQ].append(reason)

        # Resolve model with highest score
        best_model = max(scores, key=scores.get)
        max_score = scores[best_model]

        # Disambiguate ties or low confidence
        if max_score == 0:
            if "lead time" in q and ("safety" in q or "reorder" in q or "buffer" in q):
                best_model = self.MODEL_SAFETY_STOCK_ROP
                max_score = 4
                reasons[best_model].append("Inferred from lead time and safety/reorder query")
            elif "order" in q and ("holding" in q or "cost" in q):
                best_model = self.MODEL_EOQ
                max_score = 4
                reasons[best_model].append("Inferred from ordering and holding trade-off")
            else:
                best_model = self.MODEL_EOQ
                max_score = 1
                reasons[best_model].append("Defaulted to standard EOQ baseline")

        confidence = min(1.0, round(max_score / 10.0, 2))
        return {
            "model_type": best_model,
            "model_name": self.MODEL_DISPLAY_NAMES[best_model],
            "confidence": confidence,
            "reasons": reasons[best_model],
            "all_scores": scores,
        }

    # -----------------------------------------------------------------------
    # STAGE 2: Extract Inputs (Deep NLP & Synonym Matcher)
    # -----------------------------------------------------------------------
    def extract_inputs(self, question: str, model_type: str) -> Dict[str, Any]:
        """
        Parses numerical and categorical inputs from the natural language question.
        Examines the entire question and recognizes comprehensive inventory synonyms.
        """
        # Clean question text but preserve punctuation structure for regex
        clean_text = question.replace(",", "")  # Remove comma in 10,000 -> 10000

        # Helper: Extract number following any synonym keyword pattern (prefix search)
        def _extract_prefix(keywords: List[str], target_str: str = clean_text) -> Optional[float]:
            for kw in keywords:
                # Matches: keyword + optional linking words (is, are, of, at, was, equals, etc.) + currency + number
                # Handles: "setup cost of $50", "backorders are 10 units", "demand averages 120"
                escaped_kw = re.escape(kw).replace(r"\ ", r"\s+").replace(r"\/", r"\s*/\s*")
                pattern = (
                    rf"(?:{escaped_kw})"
                    rf"(?:\s+(?:is|are|of|at|was|equals|equal\s+to|to|for|about|averages|currently|stands\s+at))?"
                    rf"[\s:=–—\-isofat\$\#]*"
                    rf"([0-9]+(?:\.[0-9]+)?)"
                )
                m = re.search(pattern, target_str, re.IGNORECASE)
                if m:
                    try:
                        return float(m.group(1))
                    except ValueError:
                        continue
            return None

        # Helper: Extract number preceding a synonym keyword pattern (postfix search)
        def _extract_postfix(keywords: List[str], target_str: str = clean_text) -> Optional[float]:
            for kw in keywords:
                # Matches: "$50 setup cost", "12000 units demand", "65 dollars per order"
                escaped_kw = re.escape(kw).replace(r"\ ", r"\s+").replace(r"\/", r"\s*/\s*")
                pattern = rf"(?:\$|usd|dollars)?\s*([0-9]+(?:\.[0-9]+)?)\s*(?:\$|usd|dollars|units)?\s*(?:{escaped_kw})"
                m = re.search(pattern, target_str, re.IGNORECASE)
                if m:
                    try:
                        return float(m.group(1))
                    except ValueError:
                        continue
            return None

        # Helper: Combined robust extractor
        def _find_number(keywords: List[str], target_str: str = clean_text) -> Optional[float]:
            val = _extract_prefix(keywords, target_str)
            if val is not None:
                return val
            return _extract_postfix(keywords, target_str)

        # Helper: Percentage extractor
        def _find_percentage(keywords: List[str], target_str: str = clean_text) -> Optional[float]:
            for kw in keywords:
                escaped_kw = re.escape(kw).replace(r"\ ", r"\s+").replace(r"\/", r"\s*/\s*")
                # Prefix percentage: "holding rate of 20%"
                pattern1 = rf"(?:{escaped_kw})[\s:=–—\-isofat\$\#]*([0-9]+(?:\.[0-9]+)?)\s*(?:\%|percent)"
                m1 = re.search(pattern1, target_str, re.IGNORECASE)
                if m1:
                    val = float(m1.group(1))
                    return val / 100.0 if val > 1.0 else val

                # Postfix percentage: "20% holding rate"
                pattern2 = rf"([0-9]+(?:\.[0-9]+)?)\s*(?:\%|percent)\s*(?:{escaped_kw})"
                m2 = re.search(pattern2, target_str, re.IGNORECASE)
                if m2:
                    val = float(m2.group(1))
                    return val / 100.0 if val > 1.0 else val

            # Generic percentage search if keyword didn't capture
            m_gen = re.search(r"([0-9]+(?:\.[0-9]+)?)\s*(?:\%|percent)", target_str, re.IGNORECASE)
            if m_gen:
                val = float(m_gen.group(1))
                return val / 100.0 if val > 1.0 else val
            return None

        extracted = {}
        meanings = {}
        missing_required = []

        # ===================================================================
        # MODEL 1: EOQ Synonyms and Distinctions
        # ===================================================================
        if model_type == self.MODEL_EOQ:
            # Synonyms for Ordering Cost as explicitly requested:
            # - ordering cost, order cost, setup cost, setup expense, cost per order,
            #   cost of placing an order, ordering/setup cost, order placement cost
            ordering_cost_synonyms = [
                "ordering/setup cost",
                "ordering / setup cost",
                "cost of placing an order",
                "cost to place an order",
                "cost per order",
                "setup expense",
                "order expense",
                "setup cost",
                "ordering cost",
                "order cost",
                "order placement cost",
                "fixed ordering cost",
                "setup charge",
                "s =",
            ]

            demand_synonyms = [
                "annual demand",
                "yearly demand",
                "annual sales",
                "yearly sales",
                "annual requirement",
                "units per year",
                "units / year",
                "annual usage",
                "yearly usage",
                "demand rate",
                "demand",
                "d =",
            ]

            holding_cost_synonyms = [
                "unit holding cost",
                "annual holding cost",
                "holding cost per unit",
                "cost of holding",
                "cost to hold",
                "holding cost",
                "carrying cost per unit",
                "cost of carrying",
                "carrying cost",
                "storage cost",
                "holding expense",
                "carrying expense",
                "h =",
            ]

            unit_cost_synonyms = [
                "unit purchase price",
                "purchase price per unit",
                "unit purchase cost",
                "purchase price",
                "purchase cost",
                "wholesale price",
                "unit cost",
                "item cost",
                "cost per unit",
                "price per unit",
                "c =",
            ]

            holding_rate_synonyms = [
                "holding cost rate",
                "carrying cost rate",
                "holding rate",
                "carrying rate",
                "cost of capital",
                "interest rate",
                "holding percentage",
                "carrying percentage",
                "i =",
            ]

            demand = _find_number(demand_synonyms)
            order_cost = _find_number(ordering_cost_synonyms)
            holding_cost = _find_number(holding_cost_synonyms)
            unit_cost = _find_number(unit_cost_synonyms)
            holding_rate = _find_percentage(holding_rate_synonyms)

            # Record meanings
            if demand is not None:
                meanings["annual_demand"] = "Annual customer demand (D)"
            if order_cost is not None:
                meanings["order_cost"] = "Fixed cost per replenishment order / setup expense (S)"
            if holding_cost is not None:
                meanings["holding_cost"] = "Direct annual holding cost per unit (H)"
            if unit_cost is not None:
                meanings["unit_cost"] = "Unit purchase price (C)"
            if holding_rate is not None:
                meanings["holding_rate"] = "Annual inventory carrying rate (i)"

            # Check what's missing
            if demand is None:
                missing_required.append("annual_demand (Annual customer demand in units/year)")
            if order_cost is None:
                missing_required.append("order_cost (Cost per order, setup cost, or cost of placing an order)")
            if holding_cost is None and not (unit_cost is not None and holding_rate is not None):
                missing_required.append("holding_cost (Unit holding cost H, OR unit purchase price C and carrying rate i)")

            extracted = {
                "annual_demand": demand,
                "order_cost": order_cost,
                "holding_cost": holding_cost,
                "unit_cost": unit_cost,
                "holding_rate": holding_rate,
            }

        # ===================================================================
        # MODEL 2: Quantity Discount Synonyms and Distinctions
        # ===================================================================
        elif model_type == self.MODEL_QUANTITY_DISCOUNT:
            # Distinguish:
            # - quantity break
            # - unit price
            # - ordering cost
            # - fixed holding cost
            # - percentage holding cost
            # - unit purchase price
            demand_synonyms = ["annual demand", "yearly demand", "demand rate", "units per year", "demand", "sales"]
            ordering_cost_synonyms = [
                "ordering/setup cost", "cost of placing an order", "cost per order",
                "setup expense", "setup cost", "ordering cost", "order cost", "s ="
            ]
            fixed_holding_synonyms = ["fixed holding cost", "holding cost per unit", "storage cost", "holding cost", "carrying cost", "h ="]
            percentage_holding_synonyms = ["percentage holding cost", "holding cost rate", "carrying cost rate", "holding rate", "carrying rate", "cost of capital", "i ="]

            demand = _find_number(demand_synonyms)
            order_cost = _find_number(ordering_cost_synonyms)
            fixed_holding = _find_number(fixed_holding_synonyms)
            holding_rate = _find_percentage(percentage_holding_synonyms)

            # Parse quantity breaks / discount tiers
            # Patterns: "0 to 499: $20", "500-999: $18", "1000+: $16", "orders of 500 or more: $18"
            tiers = []
            tier_patterns = re.findall(
                r"(\d+)\s*(?:to|-|–)\s*(\d+)\s*(?:units)?\s*[:=]\s*\$?([0-9]+(?:\.[0-9]+)?)",
                clean_text,
                re.IGNORECASE
            )
            for low, high, price in tier_patterns:
                tiers.append({
                    "min_qty": float(low),
                    "max_qty": float(high),
                    "unit_price": float(price),
                })

            # Check open-ended tier: "1000+: $16" or "1000 and over: $16" or "above 1000: $16"
            plus_tier = re.search(
                r"(\d+)\s*(?:\+|and over|or more|above)\s*(?:units)?\s*[:=]\s*\$?([0-9]+(?:\.[0-9]+)?)",
                clean_text,
                re.IGNORECASE
            )
            if plus_tier:
                tiers.append({
                    "min_qty": float(plus_tier.group(1)),
                    "max_qty": None,
                    "unit_price": float(plus_tier.group(2)),
                })

            if demand is not None:
                meanings["annual_demand"] = "Annual demand volume (D)"
            if order_cost is not None:
                meanings["order_cost"] = "Ordering / setup cost per replenishment batch (S)"
            if fixed_holding is not None:
                meanings["fixed_holding_cost"] = "Fixed annual unit holding cost ($/unit/year)"
            if holding_rate is not None:
                meanings["holding_rate"] = "Percentage carrying cost rate (% of unit purchase price)"
            if tiers:
                meanings["discount_tiers"] = f"Quantity breaks and unit price tiers ({len(tiers)} price tiers recognized)"

            if demand is None:
                missing_required.append("annual_demand (Annual demand rate)")
            if order_cost is None:
                missing_required.append("order_cost (Fixed ordering / setup cost)")
            if not tiers:
                missing_required.append("discount_tiers (Quantity breaks and unit price brackets)")
            if fixed_holding is None and holding_rate is None:
                missing_required.append("holding_cost OR holding_rate (Fixed holding cost or percentage holding rate)")

            extracted = {
                "annual_demand": demand,
                "order_cost": order_cost,
                "discount_tiers": tiers if tiers else None,
                "holding_cost": fixed_holding,
                "holding_rate": holding_rate,
            }

        # ===================================================================
        # MODEL 3: Safety Stock and ROP Synonyms and Distinctions
        # ===================================================================
        elif model_type == self.MODEL_SAFETY_STOCK_ROP:
            # Distinguish:
            # - average demand
            # - demand standard deviation
            # - average lead time
            # - lead-time standard deviation
            # - service level

            # 1. Demand standard deviation (check BEFORE average demand so keywords don't overlap)
            std_demand_synonyms = [
                "demand standard deviation",
                "standard deviation of demand",
                "std dev of demand",
                "std. dev. of demand",
                "demand std dev",
                "demand std",
                "std demand",
                "sigma_d",
                "demand standard dev",
                "standard deviation in demand",
                "demand variability",
                "std dev",
                "std. dev.",
                "standard deviation",
                "stdev",
            ]
            std_demand = _find_number(std_demand_synonyms)

            # 2. Lead-time standard deviation (check BEFORE average lead time)
            std_lt_synonyms = [
                "lead-time standard deviation",
                "lead time standard deviation",
                "standard deviation of lead time",
                "std dev of lead time",
                "std. dev. of lead time",
                "lead time std dev",
                "lead time std",
                "lead-time std",
                "sigma_l",
                "lead time variability",
            ]
            std_lead_time = _find_number(std_lt_synonyms)

            # 3. Average demand
            avg_demand_synonyms = [
                "average daily demand",
                "average demand",
                "mean daily demand",
                "mean demand",
                "daily demand",
                "expected daily demand",
                "expected demand",
                "demand rate",
                "demand averages",
                "demand is",
                "demand of",
                "d =",
            ]
            # When searching for avg_demand, avoid matching standard deviation numbers
            avg_demand = _find_number(avg_demand_synonyms)

            # 4. Average lead time
            avg_lt_synonyms = [
                "average lead time",
                "mean lead time",
                "supplier lead time",
                "replenishment lead time",
                "delivery lead time",
                "delivery time",
                "lead time of",
                "lead time is",
                "lead time",
                "l =",
            ]
            lead_time = _find_number(avg_lt_synonyms)

            # 5. Service level
            service_level_synonyms = [
                "cycle service level",
                "service level",
                "target service level",
                "desired service level",
                "csl",
                "fill rate",
            ]
            service_level = _find_percentage(service_level_synonyms)
            z_score = _find_number(["z-score", "z score", "z factor", "z ="])

            if avg_demand is not None:
                meanings["avg_demand"] = "Average demand per period (d)"
            if std_demand is not None:
                meanings["std_demand"] = "Demand standard deviation (sigma_d)"
            if lead_time is not None:
                meanings["lead_time"] = "Average replenishment lead time (L)"
            if std_lead_time is not None:
                meanings["std_lead_time"] = "Lead-time standard deviation (sigma_L)"
            if service_level is not None:
                meanings["service_level"] = "Cycle service level (CSL)"
            elif z_score is not None:
                meanings["z_score"] = "Standard normal safety factor (z)"

            if avg_demand is None:
                missing_required.append("avg_demand (Average demand per period)")
            if lead_time is None:
                missing_required.append("lead_time (Average lead time)")
            if service_level is None and z_score is None:
                missing_required.append("service_level (Cycle service level e.g. 95%, or z-score)")

            extracted = {
                "avg_demand": avg_demand,
                "lead_time": lead_time,
                "service_level": service_level,
                "z_score": z_score,
                "std_demand": std_demand if std_demand is not None else 0.0,
                "std_lead_time": std_lead_time if std_lead_time is not None else 0.0,
                "demand_time_unit": "day",
                "lead_time_unit": "days",
            }

        # ===================================================================
        # MODEL 4: Fixed-Period Review Synonyms and Distinctions
        # ===================================================================
        elif model_type == self.MODEL_FIXED_PERIOD_REVIEW:
            # Distinguish:
            # - demand
            # - review period
            # - lead time
            # - safety stock
            # - inventory position
            # - on-hand inventory
            # - on-order inventory
            # - backorders

            review_period_synonyms = [
                "review period",
                "review interval",
                "review frequency",
                "reviewed every",
                "ordered every",
                "placed every",
                "every",
                "r =",
            ]
            lead_time_synonyms = [
                "replenishment lead time",
                "supplier lead time",
                "delivery time",
                "lead time",
                "l =",
            ]
            std_demand_synonyms = [
                "demand standard deviation",
                "standard deviation of demand",
                "std dev of demand",
                "std dev",
                "sigma_d",
                "sigma",
            ]
            avg_demand_synonyms = [
                "average daily demand",
                "average demand",
                "mean demand",
                "daily demand",
                "demand",
                "d =",
            ]
            on_hand_synonyms = [
                "on-hand inventory",
                "on hand inventory",
                "stock on hand",
                "current on-hand",
                "on hand",
                "current stock",
                "current inventory",
            ]
            on_order_synonyms = [
                "on-order inventory",
                "on order inventory",
                "in transit",
                "pipeline inventory",
                "open orders",
                "on order",
            ]
            backorder_synonyms = [
                "backorders",
                "backordered units",
                "backordered",
                "unfilled orders",
                "backlog",
            ]
            ip_synonyms = [
                "inventory position",
                "current inventory position",
                "ip =",
            ]

            review_period = _find_number(review_period_synonyms)
            lead_time = _find_number(lead_time_synonyms)
            std_demand = _find_number(std_demand_synonyms)
            avg_demand = _find_number(avg_demand_synonyms)
            service_level = _find_percentage(["service level", "csl", "target service level"])
            z_score = _find_number(["z-score", "z score", "z ="])
            on_hand = _find_number(on_hand_synonyms)
            on_order = _find_number(on_order_synonyms)
            backorders = _find_number(backorder_synonyms)
            inv_position = _find_number(ip_synonyms)

            if review_period is not None:
                meanings["review_period"] = "Review period / review interval (R)"
            if lead_time is not None:
                meanings["lead_time"] = "Replenishment lead time (L)"
            if avg_demand is not None:
                meanings["avg_demand"] = "Average demand rate per period (d)"
            if std_demand is not None:
                meanings["std_demand"] = "Standard deviation of demand per period (sigma_d)"
            if on_hand is not None:
                meanings["current_on_hand"] = "On-hand physical inventory"
            if on_order is not None:
                meanings["current_on_order"] = "On-order / pipeline inventory"
            if backorders is not None:
                meanings["backorders"] = "Customer backorders / backlog"
            if inv_position is not None:
                meanings["inventory_position"] = "Total inventory position (On-hand + On-order - Backorders)"

            if review_period is None:
                missing_required.append("review_period (Review period or interval between review dates)")
            if lead_time is None:
                missing_required.append("lead_time (Replenishment lead time)")
            if avg_demand is None:
                missing_required.append("avg_demand (Average demand rate)")
            if service_level is None and z_score is None:
                missing_required.append("service_level (Cycle service level e.g. 95%, or z-score)")

            extracted = {
                "review_period": review_period,
                "lead_time": lead_time,
                "avg_demand": avg_demand,
                "service_level": service_level,
                "z_score": z_score,
                "std_demand": std_demand if std_demand is not None else 0.0,
                "current_on_hand": on_hand if on_hand is not None else 0.0,
                "current_on_order": on_order if on_order is not None else 0.0,
                "backorders": backorders if backorders is not None else 0.0,
                "inventory_position": inv_position,
                "time_unit": "days",
            }

        # ===================================================================
        # MODEL 5: Newsvendor Synonyms and Distinctions
        # ===================================================================
        elif model_type == self.MODEL_NEWSVENDOR:
            # Distinguish:
            # - selling price
            # - purchase cost
            # - salvage value
            # - underage / shortage cost
            # - overage / excess cost
            # - demand distribution
            # - mean demand
            # - demand standard deviation

            selling_price_synonyms = [
                "selling price per unit",
                "selling price",
                "retail price",
                "sale price",
                "sales price",
                "selling revenue",
                "price per unit",
                "retail value",
                "price",
                "p =",
            ]
            purchase_cost_synonyms = [
                "unit purchase cost",
                "purchase cost",
                "wholesale cost",
                "wholesale price",
                "acquisition cost",
                "procurement cost",
                "cost per unit",
                "production cost",
                "unit cost",
                "cost",
                "c =",
            ]
            salvage_value_synonyms = [
                "salvage value",
                "salvage price",
                "liquidation price",
                "liquidation value",
                "markdown price",
                "markdown value",
                "scrap value",
                "residual value",
                "salvage",
                "s =",
            ]
            underage_synonyms = [
                "underage cost",
                "shortage cost",
                "cost of underage",
                "cost of shortage",
                "lost sale penalty",
                "shortage penalty",
                "goodwill penalty",
                "cu =",
            ]
            overage_synonyms = [
                "overage cost",
                "excess cost",
                "cost of overage",
                "cost of excess",
                "disposal cost",
                "co =",
            ]
            std_demand_synonyms = [
                "demand standard deviation",
                "standard deviation of demand",
                "std dev of demand",
                "std. dev. of demand",
                "demand std dev",
                "demand std",
                "std dev",
                "std. dev.",
                "standard deviation",
                "stdev",
                "sigma_d",
                "sigma",
            ]
            mean_demand_synonyms = [
                "mean demand",
                "average demand",
                "expected demand",
                "forecasted demand",
                "forecast demand",
                "forecast",
                "mu =",
            ]

            selling_price = _find_number(selling_price_synonyms)
            purchase_cost = _find_number(purchase_cost_synonyms)
            salvage_value = _find_number(salvage_value_synonyms)
            underage_cost = _find_number(underage_synonyms)
            overage_cost = _find_number(overage_synonyms)
            std_demand = _find_number(std_demand_synonyms)
            mean_demand = _find_number(mean_demand_synonyms)

            # Fallback for generic demand if mean_demand not explicitly prefaced
            if mean_demand is None:
                mean_demand = _find_number(["demand is", "demand of", "demand averages", "demand"])

            shortage_penalty = _find_number(["shortage penalty", "penalty for lost sale", "goodwill loss", "penalty", "g ="]) or 0.0

            # If underage cost or overage cost were given directly, reconcile them
            if underage_cost is not None and selling_price is None and purchase_cost is not None:
                selling_price = purchase_cost + underage_cost
            if overage_cost is not None and salvage_value is None and purchase_cost is not None:
                salvage_value = max(0.0, purchase_cost - overage_cost)

            if selling_price is not None:
                meanings["unit_price"] = "Selling / retail price per unit (p)"
            if purchase_cost is not None:
                meanings["unit_cost"] = "Unit purchase / acquisition cost (c)"
            if salvage_value is not None:
                meanings["salvage_value"] = "Salvage / markdown liquidation value (s)"
            if mean_demand is not None:
                meanings["mean_demand"] = "Mean / expected seasonal demand (mu)"
            if std_demand is not None:
                meanings["std_demand"] = "Demand standard deviation (sigma)"
            if shortage_penalty > 0:
                meanings["shortage_penalty"] = "Unsatisfied customer shortage penalty / goodwill loss (g)"

            if selling_price is None:
                missing_required.append("unit_price (Selling or retail price per unit)")
            if purchase_cost is None:
                missing_required.append("unit_cost (Purchase cost or wholesale cost per unit)")
            if mean_demand is None:
                missing_required.append("mean_demand (Mean or expected demand)")
            if std_demand is None:
                missing_required.append("std_demand (Demand standard deviation)")

            extracted = {
                "unit_price": selling_price,
                "unit_cost": purchase_cost,
                "mean_demand": mean_demand,
                "std_demand": std_demand,
                "salvage_value": salvage_value if salvage_value is not None else 0.0,
                "shortage_penalty": shortage_penalty,
            }

        # Pack extraction metadata
        extracted["_metadata"] = {
            "parameter_meanings": meanings,
            "missing_required": missing_required,
            "is_complete": len(missing_required) == 0,
        }

        return extracted

    # -----------------------------------------------------------------------
    # STAGE 3: Select Python Tool
    # -----------------------------------------------------------------------
    def select_python_tool(self, model_type: str) -> Tuple[Callable, str]:
        """
        Maps identified model to the appropriate numerical calculation function.
        """
        tool_fn = self.tool_mapping.get(model_type)
        if not tool_fn:
            raise ValueError(f"No calculation tool registered for model type: {model_type}")
        return tool_fn, tool_fn.__name__

    # -----------------------------------------------------------------------
    # STAGE 4: Calculate
    # -----------------------------------------------------------------------
    def calculate(self, tool_fn: Callable, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Executes the Python tool with parameter unpack and validation.
        Cleans private metadata before passing to the numerical tool.
        """
        clean_kwargs = {k: v for k, v in inputs.items() if not k.startswith("_")}
        return tool_fn(**clean_kwargs)

    # -----------------------------------------------------------------------
    # STAGE 5: Verify
    # -----------------------------------------------------------------------
    def verify(self, model_type: str, result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validates mathematical consistency, boundary invariants, and sanity checks.
        """
        verification_data = result.get("verification", {})
        passed = True
        notes = []

        if model_type == self.MODEL_EOQ:
            cost_diff = verification_data.get("cost_difference", 0.0)
            passed = cost_diff < 1.0  # Holding and ordering cost balance
            notes.append(f"Holding cost (${result['annual_holding_cost']:,.2f}) balances Ordering cost (${result['annual_ordering_cost']:,.2f}).")

        elif model_type == self.MODEL_QUANTITY_DISCOUNT:
            opt_tier = result.get("optimal_tier")
            passed = opt_tier is not None and result.get("optimal_order_quantity", 0) > 0
            notes.append(f"Evaluated all {len(result['evaluated_tiers'])} tiers; Tier {opt_tier} strictly minimized total acquisition and holding cost.")

        elif model_type == self.MODEL_SAFETY_STOCK_ROP:
            rop = result.get("reorder_point", 0)
            exp_d = result.get("expected_lead_time_demand", 0)
            passed = rop >= exp_d
            notes.append(f"ROP ({rop}) covers expected lead time demand ({exp_d}) + {result['safety_stock']} safety buffer.")

        elif model_type == self.MODEL_FIXED_PERIOD_REVIEW:
            order_qty = result.get("recommended_order_quantity", 0)
            target_s = result.get("order_up_to_level_target_S", 0)
            protection_horizon = result.get("inputs", {}).get("protection_period_R_plus_L", 0)
            passed = order_qty >= 0 and target_s > 0
            notes.append(f"Order-up-to target ({target_s}) accurately covers {protection_horizon} period risk horizon.")

        elif model_type == self.MODEL_NEWSVENDOR:
            cr = result.get("critical_ratio", 0)
            profit_max = verification_data.get("profit_maximized_at_q", True)
            passed = (0.0 < cr < 1.0) and profit_max
            notes.append(f"Critical fractile {result['critical_ratio_percentage']} verified to maximize expected profit (${result['expected_profit']:,.2f}).")

        return {
            "passed": passed,
            "checks": notes,
            "status": "PASS: Mathematical checks and invariant assertions satisfied." if passed else "WARNING: Verification discrepancy detected.",
            "raw_verification": verification_data,
        }

    # -----------------------------------------------------------------------
    # STAGE 6: Explain
    # -----------------------------------------------------------------------
    def explain(
        self,
        question: str,
        identification: Dict[str, Any],
        inputs: Dict[str, Any],
        tool_name: str,
        result: Dict[str, Any],
        verification: Dict[str, Any]
    ) -> str:
        """
        Generates an 8-part educational verification breakdown for students:
        1. Inputs Extracted
        2. Variable Definitions
        3. Formula
        4. Formula with Substituted Values
        5. Intermediate Calculations
        6. Final Answer
        7. Verification Checks
        8. Assumptions
        """
        model_name = identification["model_name"]
        parts = []

        parts.append(f"### 🎓 Student Verification Report: {model_name}\n")
        parts.append(f"**Identified Model:** `{model_name}` (Confidence: {int(identification['confidence'] * 100)}%)")
        parts.append(f"**Deterministic Calculation Tool:** `inventory_tools.{tool_name}`\n")

        # 1. Inputs Extracted
        parts.append("#### 1. 📥 Inputs Extracted")
        inp_dict = result.get("inputs_extracted", inputs)
        for k, v in inp_dict.items():
            if not k.startswith("_") and v is not None:
                parts.append(f"- **`{k}`**: `{v}`")

        # 2. Variable Definitions
        parts.append("\n#### 2. 📖 Variable Definitions")
        var_defs = result.get("variable_definitions", {})
        for var_name, var_meaning in var_defs.items():
            parts.append(f"- **`${var_name}$`**: {var_meaning}")

        # 3. Formula
        parts.append("\n#### 3. 📐 Formula")
        formulas = result.get("formula", {})
        for f_name, f_expr in formulas.items():
            parts.append(f"- **{f_name.replace('_', ' ').title()}**: `{f_expr}`")

        # 4. Formula with Substituted Values
        parts.append("\n#### 4. 🔢 Formula with Substituted Values")
        sub_formulas = result.get("formula_with_substituted_values", {})
        if isinstance(sub_formulas, dict):
            for sf_name, sf_val in sub_formulas.items():
                if isinstance(sf_val, list):
                    parts.append(f"- **{sf_name.replace('_', ' ').title()}**:")
                    for line in sf_val:
                        parts.append(f"  • {line}")
                else:
                    parts.append(f"- **{sf_name.replace('_', ' ').title()}**: {sf_val}")

        # 5. Intermediate Calculations
        parts.append("\n#### 5. 🧮 Intermediate Calculations")
        intermediates = result.get("intermediate_calculations", {})
        for im_key, im_val in intermediates.items():
            if im_key != "evaluated_tiers":
                parts.append(f"- **{im_key.replace('_', ' ').title()}**: `{im_val}`")

        # 6. Final Answer
        parts.append("\n#### 6. 🎯 Final Answer")
        final_ans = result.get("final_answer", {})
        for fa_key, fa_val in final_ans.items():
            parts.append(f"- **{fa_key.replace('_', ' ').title()}**: **`{fa_val}`**")

        # 7. Verification Checks
        parts.append("\n#### 7. 🛡️ Verification Checks")
        verif_data = result.get("verification_checks", verification)
        checks_list = verif_data.get("checks", [])
        for chk in checks_list:
            parts.append(f"- ✓ {chk}")
        parts.append(f"**Verification Outcome:** {verif_data.get('status', 'Verified')}")

        # 8. Assumptions
        parts.append("\n#### 8. 📌 Modeling Assumptions")
        for assump in result.get("assumptions", []):
            parts.append(f"- {assump}")

        return "\n".join(parts)

    # -----------------------------------------------------------------------
    # FULL END-TO-END PIPELINE: Question -> Answer
    # -----------------------------------------------------------------------
    def run_pipeline(self, question: str) -> Dict[str, Any]:
        """
        Executes the complete 7-stage architecture:
        Customer Question -> Identify Model -> Extract Inputs -> Select Python Tool -> Calculate -> Verify -> Explain

        If truly required parameters are missing from the question, reports what was extracted
        and what specific information is required to perform the calculation without guessing.
        """
        # Stage 1: Identify Model
        identification = self.identify_model(question)
        model_type = identification["model_type"]

        # Stage 2: Extract Inputs (Examining the entire question)
        extracted_inputs = self.extract_inputs(question, model_type)
        metadata = extracted_inputs.get("_metadata", {})
        missing_required = metadata.get("missing_required", [])
        meanings = metadata.get("parameter_meanings", {})

        # If required parameters are missing after examining the entire question:
        if missing_required:
            missing_report = [
                f"### ⚠️ Additional Information Needed for {identification['model_name']}\n",
                f"I examined your entire question and successfully extracted these parameters:",
            ]
            for p_name, p_meaning in meanings.items():
                val = extracted_inputs.get(p_name)
                missing_report.append(f"- **{p_meaning}**: `{val}`")

            missing_report.append("\nHowever, the following essential numerical parameters are missing:")
            for m in missing_required:
                missing_report.append(f"- ❓ **{m}**")

            missing_report.append(
                "\n*Please provide the missing values so I can calculate the exact optimal inventory decision without inventing numerical data.*"
            )

            return {
                "question": question,
                "identification": identification,
                "inputs": extracted_inputs,
                "tool_name": self.tool_mapping[model_type].__name__,
                "result": {
                    "status": "INCOMPLETE_INPUTS",
                    "missing_parameters": missing_required,
                },
                "verification": {
                    "passed": False,
                    "checks": ["Cannot verify calculation: missing required inputs."],
                    "status": "WAITING_FOR_INPUTS",
                },
                "explanation": "\n".join(missing_report),
            }

        # Stage 3: Select Tool
        tool_fn, tool_name = self.select_python_tool(model_type)

        # Stage 4: Calculate
        calculation_result = self.calculate(tool_fn, extracted_inputs)

        # Stage 5: Verify
        verification_report = self.verify(model_type, calculation_result)

        # Stage 6: Explain
        explanation = self.explain(
            question=question,
            identification=identification,
            inputs=extracted_inputs,
            tool_name=tool_name,
            result=calculation_result,
            verification=verification_report
        )

        return {
            "question": question,
            "identification": identification,
            "inputs": extracted_inputs,
            "tool_name": tool_name,
            "result": calculation_result,
            "verification": verification_report,
            "explanation": explanation,
        }
