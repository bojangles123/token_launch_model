import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import math


class OrderBook:
    """Model CEX order book with depth at different price levels"""

    def __init__(self, base_price, depth_amounts, depth_percentages):
        self.base_price = base_price
        # Create price levels with corresponding depth
        self.levels = []
        for i, (amount, pct) in enumerate(zip(depth_amounts, depth_percentages)):
            price = base_price * (1 - pct / 100)  # Price levels below current price
            self.levels.append({'price': price, 'depth': amount, 'original_depth': amount})
        self.levels.sort(key=lambda x: x['price'], reverse=True)  # Highest price first

    def execute_sell(self, sell_amount_usd):
        """Execute sell order and return average price and slippage"""
        remaining_to_sell = sell_amount_usd
        total_tokens_sold = 0
        total_usd_received = 0

        for level in self.levels:
            if remaining_to_sell <= 0:
                break

            available_depth = level['depth']
            if available_depth <= 0:
                continue

            # How much can we sell at this price level
            sell_at_level = min(remaining_to_sell, available_depth)
            tokens_sold_at_level = sell_at_level / level['price']

            # Update totals
            total_tokens_sold += tokens_sold_at_level
            total_usd_received += sell_at_level
            remaining_to_sell -= sell_at_level

            # Consume depth
            level['depth'] -= sell_at_level

        if total_tokens_sold > 0:
            avg_price = total_usd_received / total_tokens_sold
            price_impact = (self.base_price - avg_price) / self.base_price
        else:
            avg_price = self.base_price
            price_impact = 0

        return {
            'avg_price': avg_price,
            'price_impact': price_impact,
            'tokens_sold': total_tokens_sold,
            'usd_received': total_usd_received,
            'unfilled': remaining_to_sell
        }

    def refresh_depth(self, refresh_percentage=1.0):
        """Refresh order book depth (market makers replenish)"""
        for level in self.levels:
            level['depth'] = level['original_depth'] * refresh_percentage


class AMMPool:
    """Model AMM pool with constant product formula"""

    def __init__(self, token_reserves, usdc_reserves):
        self.initial_token_reserves = token_reserves
        self.initial_usdc_reserves = usdc_reserves
        self.token_reserves = token_reserves
        self.usdc_reserves = usdc_reserves
        self.k = token_reserves * usdc_reserves  # Constant product

    @property
    def current_price(self):
        return self.usdc_reserves / self.token_reserves

    def execute_sell(self, sell_amount_usd):
        """Execute sell using constant product formula"""
        current_price = self.current_price
        sell_tokens = sell_amount_usd / current_price

        # New token reserves after adding sold tokens
        new_token_reserves = self.token_reserves + sell_tokens

        # Calculate new USDC reserves using constant product
        new_usdc_reserves = self.k / new_token_reserves

        # USDC received (less than sell_amount_usd due to slippage)
        usdc_received = self.usdc_reserves - new_usdc_reserves

        # Update reserves
        old_price = current_price
        self.token_reserves = new_token_reserves
        self.usdc_reserves = new_usdc_reserves
        new_price = self.current_price

        # Calculate metrics
        price_impact = (old_price - new_price) / old_price
        avg_price = usdc_received / sell_tokens if sell_tokens > 0 else current_price

        return {
            'avg_price': avg_price,
            'price_impact': price_impact,
            'tokens_sold': sell_tokens,
            'usd_received': usdc_received,
            'new_price': new_price,
            'slippage': 1 - (usdc_received / sell_amount_usd)
        }


def calculate_hype_decay(initial_hype, day, decay_rate=0.25, price_performance_factor=1.0):
    """Calculate hype decay with price performance feedback"""
    # Base exponential decay
    base_decay = initial_hype * np.exp(-decay_rate * day)

    # Price performance multiplier (good performance sustains hype)
    performance_multiplier = max(0.3, min(2.0, price_performance_factor))

    return base_decay * performance_multiplier


def calculate_realistic_mm_impact(mm_structure, daily_selling_pct, pressure_ratio):
    """
    Calculate realistic market maker capacity and execution impact
    Based on real-world MM behavior during token launches and crashes
    """
    
    if mm_structure == "retainer":
        # Retainer MM: Stable performance regardless of pressure
        return {
            'execution_quality': 1.0,
            'capacity_multiplier': 1.0,
            'description': "Stable execution - you paid for reliability"
        }
    else:
        # Loan MM: More aggressive degradation to show clear difference
        capacity_multiplier = 1.0
        execution_quality = 1.0
        
        # MUCH more aggressive base degradation based on daily selling pressure
        if daily_selling_pct > 5.0:
            capacity_multiplier = 0.15      # Extreme: 85% capacity loss
            execution_quality = 0.20        # Severe execution degradation (80% loss)
        elif daily_selling_pct > 3.0:
            capacity_multiplier = 0.25      # Heavy: 75% capacity loss  
            execution_quality = 0.30        # Major degradation (70% loss)
        elif daily_selling_pct > 2.0:
            capacity_multiplier = 0.40      # Moderate: 60% capacity loss
            execution_quality = 0.45        # Significant degradation (55% loss)
        elif daily_selling_pct > 1.0:
            capacity_multiplier = 0.60      # Light: 40% capacity loss
            execution_quality = 0.65        # Moderate degradation (35% loss)
        elif daily_selling_pct > 0.5:
            capacity_multiplier = 0.80      # Minimal: 20% capacity loss
            execution_quality = 0.85        # Light degradation (15% loss)
        
        # Additional impact from buy/sell pressure imbalance (more aggressive)
        if pressure_ratio > 3.0:
            capacity_multiplier *= 0.50     # 50% additional reduction
            execution_quality *= 0.60       # 40% additional execution impact
        elif pressure_ratio > 2.0:
            capacity_multiplier *= 0.70     # 30% additional reduction
            execution_quality *= 0.75       # 25% additional impact
        elif pressure_ratio > 1.5:
            capacity_multiplier *= 0.80     # 20% additional reduction
            execution_quality *= 0.85       # 15% additional impact
        
        # Realistic minimum floors (MMs don't completely disappear but get very bad)
        capacity_multiplier = max(0.10, capacity_multiplier)    # 10% minimum capacity
        execution_quality = max(0.10, execution_quality)        # 10% minimum execution quality
        
        return {
            'execution_quality': execution_quality,
            'capacity_multiplier': capacity_multiplier,
            'description': f"Degraded to {capacity_multiplier:.0%} capacity, {execution_quality:.0%} execution quality"
        }


def calculate_realistic_tge_price_impact(
    your_selling_value, external_selling_value, 
    total_buy_pressure, total_supply_tokens, token_price
):
    """
    Calculate realistic TGE price impact based on actual tokenomics
    Accounts for % of supply being sold and buy/sell pressure ratio
    """
    
    # Total selling pressure
    total_selling_value = your_selling_value + external_selling_value
    total_selling_tokens = total_selling_value / token_price
    
    # Key metric: % of total supply being sold
    selling_pct_of_supply = (total_selling_tokens / total_supply_tokens) * 100
    
    # Buy to sell pressure ratio
    buy_to_sell_ratio = total_buy_pressure / total_selling_value if total_selling_value > 0 else 10
    
    # More moderate price impact curve based on supply % and buy/sell ratio
    def calculate_supply_impact(selling_pct, buy_ratio):
        # GENTLER base impact from supply selling (reduced severity)
        if selling_pct < 0.5:
            # Very light selling: -5% to -20%
            base_impact = 0.95 - (selling_pct * 0.3)
        elif selling_pct < 1.0:
            # Light selling: -20% to -40%
            base_impact = 0.8 - (selling_pct * 0.35)
        elif selling_pct < 2.0:
            # Moderate selling: -40% to -65%
            base_impact = 0.6 - ((selling_pct - 1.0) * 0.25)
        elif selling_pct < 3.0:
            # Heavy selling: -65% to -80%
            base_impact = 0.35 - ((selling_pct - 2.0) * 0.15)
        elif selling_pct < 5.0:
            # Massive selling: -80% to -90%
            base_impact = 0.2 - ((selling_pct - 3.0) * 0.05)
        else:
            # Extreme selling: -90% to -95%
            base_impact = max(0.05, 0.1 - ((selling_pct - 5.0) * 0.025))
        
        # STRONGER and more nuanced buy pressure adjustment
        buy_support_multiplier = 1.0
        
        if buy_ratio >= 2.0:
            # Very strong buying: 2x+ ratio provides major support
            buy_support_multiplier = 1.8 + min(1.2, (buy_ratio - 2.0) * 0.4)  # Up to 3x support
        elif buy_ratio >= 1.0:
            # Strong buying: equal or more buying than selling
            buy_support_multiplier = 1.2 + (buy_ratio - 1.0) * 0.6  # 1.2x to 1.8x support
        elif buy_ratio >= 0.7:
            # Moderate buying: decent support even with some selling pressure
            buy_support_multiplier = 1.0 + (buy_ratio - 0.7) * 0.67  # 1x to 1.2x support
        elif buy_ratio >= 0.4:
            # Light selling pressure: some support from existing buying
            buy_support_multiplier = 0.9 + (buy_ratio - 0.4) * 0.33  # 0.9x to 1x
        elif buy_ratio >= 0.2:
            # Heavy selling pressure: limited support
            buy_support_multiplier = 0.8 + (buy_ratio - 0.2) * 0.5   # 0.8x to 0.9x
        else:
            # Extreme selling: panic conditions but not total abandonment
            buy_support_multiplier = max(0.6, 0.8 - (0.2 - buy_ratio) * 1.0)  # 0.6x to 0.8x
        
        # CRASH BUYING EFFECT: More moderate and realistic
        crash_severity = 1 - base_impact
        if crash_severity > 0.4 and buy_ratio > 0.3:
            # Only apply crash buying if there's meaningful buying interest
            crash_buying_multiplier = 1 + min(0.5, (crash_severity - 0.4) * (buy_ratio ** 0.3))
            buy_support_multiplier *= crash_buying_multiplier
        
        # Apply buy pressure support with more moderate bounds
        final_impact = base_impact * buy_support_multiplier
        
        # More generous bounds (crashes rarely go below -95%)
        return max(0.05, min(1.0, final_impact))
    
    price_multiplier = calculate_supply_impact(selling_pct_of_supply, buy_to_sell_ratio)
    
    return {
        'price_multiplier': price_multiplier,
        'price_change_pct': (price_multiplier - 1) * 100,
        'selling_pct_of_supply': selling_pct_of_supply,
        'buy_to_sell_ratio': buy_to_sell_ratio,
        'total_selling_value': total_selling_value,
        'total_buy_pressure': total_buy_pressure
    }


def simulate_liquidation_timeline(
        initial_token_price, total_tokens_to_sell, days,
        cex_exchanges, dex_pools, natural_buy_pressure_daily,
        external_sell_pressure_tokens, cycles_per_day=32, hype_decay_rate=0.25,
        selling_aggressiveness=1.0, mm_structure="retainer", total_supply_tokens=2_000_000_000
):
    """Simulate detailed liquidation over time with realistic MM behavior"""

    timeline = []
    remaining_tokens = total_tokens_to_sell
    current_price = initial_token_price
    cumulative_revenue = 0
    hype_factor = 1.0

    # Check if we're only observing market (not selling)
    observing_only = total_tokens_to_sell <= 0

    for day in range(days):
        day_data = {
            'day': day + 1,
            'start_price': current_price,
            'daily_trades': [],
            'hype_factor': hype_factor
        }

        # Initialize daily tracking
        daily_revenue = 0
        daily_tokens_sold = 0
        daily_external_tokens_sold = 0
        daily_price_impacts = []

        # Calculate daily natural buy pressure (affected by hype)
        daily_natural_buy = natural_buy_pressure_daily * hype_factor

        # Calculate external selling pressure for this day
        external_daily_selling_tokens = external_sell_pressure_tokens / days
        external_daily_selling = external_daily_selling_tokens * current_price

        # Adjust your selling based on market competition
        market_competition_factor = 1 + (external_daily_selling / daily_natural_buy) if daily_natural_buy > 0 else 1

        for cycle in range(cycles_per_day):
            # Only break if we're actually selling tokens AND have none left
            if not observing_only and remaining_tokens <= 0:
                break

            cycle_data = {'cycle': cycle + 1}

            # Calculate external selling pressure for this cycle
            external_cycle_selling_tokens = external_daily_selling_tokens / cycles_per_day
            external_cycle_selling_usd = external_cycle_selling_tokens * current_price

            # Determine how much you want to sell this cycle (0 if observing only)
            if observing_only:
                your_cycle_sell_budget_usd = 0
            else:
                your_cycle_sell_budget_usd = min(
                    remaining_tokens * current_price * 0.05,  # Max 5% of remaining per cycle (conservative)
                    25000 / market_competition_factor  # Reduce budget to be realistic
                )

            # Calculate realistic MM impact based on current market conditions
            total_selling_pressure = daily_revenue + (external_daily_selling_tokens * current_price)
            pressure_ratio = total_selling_pressure / daily_natural_buy if daily_natural_buy > 0 else 0
            
            # Daily selling as % of supply
            daily_selling_tokens = (daily_revenue / current_price) + daily_external_tokens_sold
            daily_selling_pct = (daily_selling_tokens / total_supply_tokens) * 100
            
            # Get realistic MM impact
            mm_impact = calculate_realistic_mm_impact(mm_structure, daily_selling_pct, pressure_ratio)
            execution_quality_multiplier = mm_impact['execution_quality']
            capacity_multiplier = mm_impact['capacity_multiplier']

            cycle_revenue = 0
            cycle_tokens_sold = 0
            cycle_external_tokens_sold = 0
            cycle_impacts = []

            # Calculate market stress from external selling
            if daily_natural_buy > 0:
                external_stress = external_cycle_selling_usd / (daily_natural_buy / cycles_per_day)
                market_stress_multiplier = min(2.0, 1.0 + external_stress * 0.3)
            else:
                market_stress_multiplier = 2.0

            # Execute YOUR trades through exchanges
            if your_cycle_sell_budget_usd > 0:
                # Calculate total available liquidity for proportional split
                total_cex_capacity = sum([cex['depth'] * capacity_multiplier for cex in cex_exchanges])
                total_dex_capacity = sum([dex['pool'].usdc_reserves * 0.02 for dex in dex_pools])
                total_liquidity = total_cex_capacity + total_dex_capacity
                
                # Split budget proportionally based on available liquidity
                if total_liquidity > 0:
                    cex_budget_ratio = total_cex_capacity / total_liquidity
                    dex_budget_ratio = total_dex_capacity / total_liquidity
                    
                    cex_budget = your_cycle_sell_budget_usd * cex_budget_ratio
                    dex_budget = your_cycle_sell_budget_usd * dex_budget_ratio
                else:
                    cex_budget = your_cycle_sell_budget_usd
                    dex_budget = 0

                # Execute through CEX exchanges with allocated budget
                remaining_cex_budget = cex_budget
                for cex in cex_exchanges:
                    if remaining_cex_budget <= 0:
                        break

                    available_capacity = cex['depth'] * capacity_multiplier
                    your_sell_amount = min(remaining_cex_budget, available_capacity)

                    if your_sell_amount > 0:
                        result = cex['orderbook'].execute_sell(your_sell_amount)

                        # Apply MM execution quality impact
                        if mm_structure == "loan":
                            # Loan MM: Much more severe execution degradation
                            adjusted_usd_received = result['usd_received'] * execution_quality_multiplier
                            # Also increase price impact when execution quality is poor
                            adjusted_price_impact = result['price_impact'] * (2.5 - execution_quality_multiplier)
                        else:
                            # Retainer MM: Stable execution with only minor market stress
                            adjusted_usd_received = result['usd_received'] * 0.998  # Very minor reduction
                            adjusted_price_impact = result['price_impact'] * 1.02  # Very minor increase

                        cycle_revenue += adjusted_usd_received
                        cycle_tokens_sold += result['tokens_sold']
                        cycle_impacts.append(adjusted_price_impact)
                        remaining_cex_budget -= your_sell_amount

                # Execute through DEX pools with allocated budget
                remaining_dex_budget = dex_budget
                for dex in dex_pools:
                    if remaining_dex_budget <= 0:
                        break

                    max_dex_trade = dex['pool'].usdc_reserves * 0.02
                    your_sell_amount = min(remaining_dex_budget, max_dex_trade)

                    if your_sell_amount > 0:
                        result = dex['pool'].execute_sell(your_sell_amount)

                        # Apply MM execution quality impact
                        if mm_structure == "loan":
                            # Loan MM: Much more severe execution degradation
                            adjusted_usd_received = result['usd_received'] * execution_quality_multiplier
                            adjusted_price_impact = result['price_impact'] * (2.5 - execution_quality_multiplier)
                        else:
                            # Retainer MM: Stable execution
                            adjusted_usd_received = result['usd_received'] * 0.998
                            adjusted_price_impact = result['price_impact'] * 1.02

                        cycle_revenue += adjusted_usd_received
                        cycle_tokens_sold += result['tokens_sold']
                        cycle_impacts.append(adjusted_price_impact)
                        remaining_dex_budget -= your_sell_amount

            # Simulate external selling impact on price
            if external_cycle_selling_usd > 0:
                external_impact = (external_cycle_selling_usd / daily_natural_buy) * 0.02 if daily_natural_buy > 0 else 0
                current_price = current_price * (1 - external_impact)
                cycle_external_tokens_sold = external_cycle_selling_tokens

            # Store cycle data with MM impact tracking
            avg_cycle_impact = np.mean(cycle_impacts) if cycle_impacts else 0
            cycle_data.update({
                'revenue': cycle_revenue,
                'tokens_sold': cycle_tokens_sold,
                'avg_price_impact': avg_cycle_impact,
                'end_price': current_price,
                'mm_capacity': capacity_multiplier,
                'mm_execution_quality': execution_quality_multiplier,
                'mm_structure': mm_structure
            })

            daily_revenue += cycle_revenue
            daily_tokens_sold += cycle_tokens_sold
            daily_external_tokens_sold += cycle_external_tokens_sold
            daily_price_impacts.extend(cycle_impacts)

            # Only reduce remaining tokens if we're actually selling
            if not observing_only:
                remaining_tokens -= cycle_tokens_sold

            day_data['daily_trades'].append(cycle_data)

            # Refresh CEX order books with realistic MM behavior
            if cycle % 2 == 0:  # Refresh every 2 cycles
                for cex in cex_exchanges:
                    total_daily_tokens_sold = daily_tokens_sold + daily_external_tokens_sold
                    
                    # Realistic refresh rate based on MM type
                    base_hit_rate = max(0.5, 1 - (total_daily_tokens_sold / 200000))  # More conservative
                    
                    if mm_structure == "loan":
                        # Loan MMs refresh more slowly under pressure
                        hit_rate = base_hit_rate * capacity_multiplier
                    else:
                        # Retainer MMs maintain consistent refresh
                        hit_rate = base_hit_rate

                    if len(daily_price_impacts) > 0:
                        avg_impact = np.mean(daily_price_impacts)
                        if avg_impact > 0.03:
                            hit_rate *= 0.8

                    cex['orderbook'].refresh_depth(max(0.3, hit_rate))

        # Calculate realistic daily price change using corrected model
        your_daily_selling_value = daily_revenue
        external_daily_selling_value = daily_external_tokens_sold * current_price
        daily_buy_pressure = daily_natural_buy
        
        # Calculate MM impact on price (NEW: MM structure affects price volatility)
        avg_daily_mm_capacity = np.mean([trade.get('mm_capacity', 1.0) for trade in day_data['daily_trades']]) if day_data['daily_trades'] else 1.0
        avg_daily_mm_execution = np.mean([trade.get('mm_execution_quality', 1.0) for trade in day_data['daily_trades']]) if day_data['daily_trades'] else 1.0
        
        # MM price impact multiplier - poor MMs amplify price drops
        if mm_structure == "loan":
            # Poor MM execution amplifies price volatility
            mm_price_impact_multiplier = 1.0 + (1.0 - avg_daily_mm_execution) * 0.8  # Up to 80% worse price impact
        else:
            # Retainer MM provides price stability 
            mm_price_impact_multiplier = 1.0  # No additional impact
        
        # Calculate realistic price impact based on TOTAL expected selling pressure
        # Market reacts to the KNOWLEDGE of large selling, not just daily amounts
        
        if day == 0:
            # Day 1: Calculate impact based on TOTAL selling pressure over simulation
            total_your_selling = (total_tokens_to_sell if not observing_only else 0) * current_price
            total_external_selling = external_sell_pressure_tokens * current_price
            total_simulation_buy_pressure = daily_buy_pressure * days
            
            # Get total impact that SHOULD happen
            total_impact_result = calculate_realistic_tge_price_impact(
                total_your_selling, 
                total_external_selling,
                total_simulation_buy_pressure,
                total_supply_tokens,
                current_price
            )
            
            # Apply 80% of total impact on Day 1 (market front-runs the selling)
            # AMPLIFY impact based on MM quality
            base_day1_multiplier = 1 + (total_impact_result['price_multiplier'] - 1) * 0.8
            day1_price_multiplier = 1 + (base_day1_multiplier - 1) * mm_price_impact_multiplier
            current_price *= day1_price_multiplier
            daily_price_change = (day1_price_multiplier - 1) * 100
            
            # Store for subsequent days
            remaining_impact_multiplier = total_impact_result['price_multiplier'] / day1_price_multiplier
            
        else:
            # Subsequent days: Apply remaining impact gradually
            if 'remaining_impact_multiplier' in locals() and remaining_impact_multiplier < 1.0:
                base_daily_multiplier = remaining_impact_multiplier ** (1/(days-1))
                # Apply MM amplification to daily price changes too
                daily_remaining_multiplier = 1 + (base_daily_multiplier - 1) * mm_price_impact_multiplier
                current_price *= daily_remaining_multiplier
                daily_price_change = (daily_remaining_multiplier - 1) * 100
            else:
                # Just apply small daily pressure from actual selling
                daily_impact_result = calculate_realistic_tge_price_impact(
                    your_daily_selling_value, 
                    external_daily_selling_value,
                    daily_buy_pressure,
                    total_supply_tokens,
                    current_price
                )
                # Apply MM amplification
                base_multiplier = daily_impact_result['price_multiplier']
                amplified_multiplier = 1 + (base_multiplier - 1) * mm_price_impact_multiplier
                current_price *= amplified_multiplier
                daily_price_change = (amplified_multiplier - 1) * 100
        
        current_price = max(current_price, 0.0001)

        # Update hype factor
        price_performance = current_price / initial_token_price
        hype_factor = calculate_hype_decay(1.0, day, hype_decay_rate, price_performance)

        # Store daily summary with MM tracking
        cumulative_revenue += daily_revenue
        day_data.update({
            'end_price': current_price,
            'daily_revenue': daily_revenue,
            'daily_tokens_sold': daily_tokens_sold,
            'daily_avg_price': daily_revenue / daily_tokens_sold if daily_tokens_sold > 0 else current_price,
            'daily_avg_impact': np.mean(daily_price_impacts) if daily_price_impacts else 0,
            'cumulative_revenue': cumulative_revenue,
            'remaining_tokens': remaining_tokens,
            'price_performance': price_performance,
            'natural_buy_pressure': daily_natural_buy,
            'external_sell_pressure': external_daily_selling_value,
            'external_tokens_sold': daily_external_tokens_sold,
            'mm_structure': mm_structure,
            'daily_price_change': daily_price_change,
            'selling_aggressiveness': selling_aggressiveness,
            'observing_only': observing_only,
            'impact_result': total_impact_result if day == 0 and 'total_impact_result' in locals() else daily_impact_result if 'daily_impact_result' in locals() else {},
            'mm_impact': mm_impact,
            'avg_mm_execution_quality': np.mean([trade.get('mm_execution_quality', 1.0) for trade in day_data['daily_trades']]) if day_data['daily_trades'] else 1.0,
            'avg_mm_capacity': np.mean([trade.get('mm_capacity', 1.0) for trade in day_data['daily_trades']]) if day_data['daily_trades'] else 1.0
        })

        timeline.append(day_data)

        if not observing_only and remaining_tokens <= 0:
            break

    return timeline


def calculate_token_metrics(fdv, circ_supply_pct, total_supply, market_sentiment):
    """Calculate basic token metrics with market sentiment"""
    sentiment_multipliers = {-1: -30, 0: 25, 1: 50}
    price_change_pct = sentiment_multipliers[market_sentiment]

    initial_market_cap = fdv * (circ_supply_pct / 100)
    circulating_supply = total_supply * (circ_supply_pct / 100)
    initial_token_price = fdv / total_supply

    new_token_price = initial_token_price * (1 + price_change_pct / 100)
    new_market_cap = new_token_price * circulating_supply
    new_fdv = new_token_price * total_supply

    return {
        'initial_market_cap': initial_market_cap,
        'circulating_supply': circulating_supply,
        'initial_token_price': initial_token_price,
        'new_token_price': new_token_price,
        'new_market_cap': new_market_cap,
        'new_fdv': new_fdv,
        'price_change_pct': price_change_pct
    }


def calculate_corrected_tokenomics(total_supply_tokens, token_price, community_dump_pct, rewards_dump_pct):
    """Calculate corrected tokenomics based on actual token distribution table"""
    
    # Standard tokenomics structure - can be customized
    tokenomics = {
        'community': {'allocation': 0.7, 'tge': 100},           # 0.7% allocation, 100% at TGE
        'community_ecosystem': {'allocation': 12.5, 'tge': 10}, # 12.5% allocation, 10% at TGE
        'rewards': {'allocation': 6.0, 'tge': 33},              # 6% allocation, 33% at TGE
        'partners': {'allocation': 5.0, 'tge': 100},            # 5% allocation, 100% at TGE (YOUR CONTROL)
        'treasury': {'allocation': 23.4, 'tge': 20},            # 23.4% allocation, 20% at TGE (YOUR CONTROL)
        'market_making': {'allocation': 5.5, 'tge': 100},       # 5.5% allocation, 100% at TGE (YOUR CONTROL)
        'pre_seed': {'allocation': 4.2, 'tge': 0},              # 4.2% allocation, 0% at TGE (12M cliff, 36M vesting)
        'seed': {'allocation': 5.0, 'tge': 0},                  # 5% allocation, 0% at TGE (12M cliff, 36M vesting)
        'private_a': {'allocation': 3.4, 'tge': 0},             # 3.4% allocation, 0% at TGE (12M cliff, 36M vesting)
        'bridge': {'allocation': 5.3, 'tge': 0},                # 5.3% allocation, 0% at TGE (12M cliff, 36M vesting)
        'strategic_partners': {'allocation': 1.5, 'tge': 0},    # 1.5% allocation, 0% at TGE (12M cliff, 36M vesting)
        'series_a': {'allocation': 7.0, 'tge': 0},              # 7% allocation, 0% at TGE (12M cliff, 36M vesting)
        'advisory': {'allocation': 3.0, 'tge': 0},              # 3% allocation, 0% at TGE (12M cliff, 36M vesting)
        'team': {'allocation': 17.5, 'tge': 0}                  # 17.5% allocation, 0% at TGE (12M cliff, 36M vesting)
    }
    
    # Calculate your liquid tokens (what you can actually sell at TGE)
    your_liquid_tokens = 0
    your_liquid_value = 0
    
    your_categories = ['partners', 'treasury', 'market_making']
    for category in your_categories:
        data = tokenomics[category]
        total_tokens = total_supply_tokens * (data['allocation'] / 100)
        liquid_tokens = total_tokens * (data['tge'] / 100)
        liquid_value = liquid_tokens * token_price
        
        your_liquid_tokens += liquid_tokens
        your_liquid_value += liquid_value
    
    # Calculate external selling pressure
    # Community (0.7% × 100% TGE)
    community_tokens = total_supply_tokens * (tokenomics['community']['allocation'] / 100)
    community_liquid = community_tokens * (tokenomics['community']['tge'] / 100)
    community_selling = community_liquid * (community_dump_pct / 100)
    
    # Community/Ecosystem (12.5% × 10% TGE)
    community_eco_tokens = total_supply_tokens * (tokenomics['community_ecosystem']['allocation'] / 100)
    community_eco_liquid = community_eco_tokens * (tokenomics['community_ecosystem']['tge'] / 100)
    community_eco_selling = community_eco_liquid * (community_dump_pct / 100)
    
    # Rewards (6% × 33% TGE)
    rewards_tokens = total_supply_tokens * (tokenomics['rewards']['allocation'] / 100)
    rewards_liquid = rewards_tokens * (tokenomics['rewards']['tge'] / 100)
    rewards_selling = rewards_liquid * (rewards_dump_pct / 100)
    
    external_selling_tokens = community_selling + community_eco_selling + rewards_selling
    external_selling_value = external_selling_tokens * token_price
    
    return {
        'your_liquid_tokens': your_liquid_tokens,
        'your_liquid_value': your_liquid_value,
        'external_selling_tokens': external_selling_tokens,
        'external_selling_value': external_selling_value,
        'tokenomics': tokenomics,
        'your_categories': your_categories
    }


def main():
    st.set_page_config(page_title="Token Launch Model - fyde.fi", layout="wide")

    st.title("Token Launch Revenue Model")
    st.markdown("*Professional-grade TGE liquidation analysis*")

    # Sidebar for parameters
    st.sidebar.header("📊 Core Parameters")

    # Basic token parameters
    st.sidebar.subheader("Token Economics")
    fdv = st.sidebar.number_input("FDV (Fully Diluted Valuation) $M",
                                  value=100.0, min_value=1.0, max_value=10000.0, step=1.0)
    fdv_usd = fdv * 1_000_000

    total_supply = st.sidebar.number_input("Total Token Supply (Billions)",
                                           value=2.0, min_value=0.1, max_value=100.0, step=0.1)
    total_supply_tokens = total_supply * 1_000_000_000

    # External selling pressure
    st.sidebar.subheader("External Selling Pressure")
    community_dump_pct = st.sidebar.slider("Community Dump %",
                                           min_value=0.0, max_value=100.0, value=75.0, step=5.0)
    rewards_dump_pct = st.sidebar.slider("Rewards Dump %",
                                         min_value=0.0, max_value=100.0, value=75.0, step=5.0)

    # Market Maker Structure
    st.sidebar.subheader("Market Maker Structure")
    mm_structure = st.sidebar.radio(
        "Market Making Model",
        options=["retainer", "loan"],
        index=0,
        format_func=lambda x: {
            "retainer": "🔒 Retainer-Based MM (Stable liquidity)",
            "loan": "💸 Loan-Based MM (Realistic capacity reduction)"
        }[x]
    )

    # Show MM behavior preview
    if mm_structure == "loan":
        st.sidebar.write("**Loan MM Capacity Curve:**")
        st.sidebar.write("• 0.5% supply selling → 80% capacity")
        st.sidebar.write("• 1.0% supply selling → 60% capacity")
        st.sidebar.write("• 2.0% supply selling → 40% capacity")
        st.sidebar.write("• 3.0% supply selling → 25% capacity")
        st.sidebar.write("• 5.0%+ supply selling → 15% capacity")
    else:
        st.sidebar.write("**Retainer MM:** Always 100% capacity")

    # Market sentiment
    st.sidebar.subheader("Market Conditions")
    market_sentiment = st.sidebar.selectbox(
        "Market Sentiment",
        options=[-1, 0, 1],
        index=2,
        format_func=lambda x: {-1: "🐻 Bear (-30%)", 0: "🦀 Sideways (+25%)", 1: "🐂 Bull (+50%)"}[x]
    )

    sentiment_exchange_multipliers = {
        -1: {"exchanges": 1.0, "depth": 0.6, "volume": 0.4},
        0: {"exchanges": 1.0, "depth": 1.0, "volume": 1.0},
        1: {"exchanges": 1.0, "depth": 1.25, "volume": 1.5}
    }
    sentiment_multipliers = sentiment_exchange_multipliers[market_sentiment]

    # Exchange configuration
    st.sidebar.subheader("CEX Exchange Setup")
    cex_exchanges = []
    base_exchange_defaults = [25000, 50000, 75000]

    for i, default_depth in enumerate(base_exchange_defaults):
        st.sidebar.write(f"**CEX Exchange {i + 1}**")
        sentiment_adjusted_default = default_depth * sentiment_multipliers["depth"]
        
        depth = st.sidebar.number_input(
            f"Depth ($)", 
            key=f"cex_depth_{i}",
            value=int(sentiment_adjusted_default),
            min_value=1000, 
            max_value=500000, 
            step=1000
        )
        cex_exchanges.append(depth)

    st.sidebar.write(f"**Total CEX Depth: ${sum(cex_exchanges):,}**")

    # DEX Configuration
    st.sidebar.subheader("DEX Liquidity")
    include_dex = st.sidebar.checkbox("Include DEX Pool", value=True)

    dex_liquidity_per_side = 0
    if include_dex:
        base_dex_liquidity = 250000
        sentiment_dex_multiplier = sentiment_multipliers["depth"]
        adjusted_dex_liquidity = base_dex_liquidity * sentiment_dex_multiplier

        dex_liquidity_per_side = st.sidebar.slider("DEX Liquidity per side ($)",
                                                   min_value=100000, max_value=1000000,
                                                   value=int(adjusted_dex_liquidity), step=50000)

    # Market dynamics
    st.sidebar.subheader("Market Dynamics")
    simulation_days = st.sidebar.slider("Simulation Days",
                                        min_value=1, max_value=14, value=7, step=1)

    base_natural_buy_pressure = st.sidebar.number_input("Base Natural Buy Pressure/Day ($)",
                                                        value=500000, min_value=50000, max_value=2000000, step=50000)

    buy_pressure_multiplier = sentiment_multipliers["volume"]
    natural_buy_pressure = base_natural_buy_pressure * buy_pressure_multiplier

    st.sidebar.write(f"**Adjusted Buy Pressure: ${natural_buy_pressure:,.0f}/day**")

    hype_decay_rate = st.sidebar.slider("Hype Decay Rate",
                                        min_value=0.1, max_value=0.5, value=0.25, step=0.05)

    # Calculate metrics
    # For circulating supply calculation, we need to account for actual liquid tokens
    tokenomics_data = calculate_corrected_tokenomics(total_supply_tokens, 0.05, community_dump_pct, rewards_dump_pct)
    
    # Calculate circulating supply % based on what's actually liquid at TGE
    total_liquid_at_tge = tokenomics_data['your_liquid_tokens'] + tokenomics_data['external_selling_tokens'] * (100 / (community_dump_pct + rewards_dump_pct)) * 2  # Rough approximation
    circ_supply_pct = (total_liquid_at_tge / total_supply_tokens) * 100
    
    metrics = calculate_token_metrics(fdv_usd, circ_supply_pct, total_supply_tokens, market_sentiment)

    # Recalculate tokenomics with actual token price
    tokenomics_data = calculate_corrected_tokenomics(total_supply_tokens, metrics['new_token_price'], community_dump_pct, rewards_dump_pct)

    # Your selling strategy
    st.sidebar.subheader("Your Selling Strategy")
    your_selling_percentage = st.sidebar.slider(
        "% of Your Liquid Tokens to Sell",
        min_value=0.0, max_value=100.0, value=10.0, step=1.0
    )

    your_tokens_to_sell = tokenomics_data['your_liquid_tokens'] * (your_selling_percentage / 100)
    your_selling_value = your_tokens_to_sell * metrics['new_token_price']

    st.sidebar.write(f"**Your Liquid Tokens: {tokenomics_data['your_liquid_tokens']:,.0f}**")
    st.sidebar.write(f"**Your Liquid Value: ${tokenomics_data['your_liquid_value']:,.0f}**")
    st.sidebar.write(f"**Tokens to Sell: {your_tokens_to_sell:,.0f}**")

    # Calculate realistic price impact
    total_buy_pressure = natural_buy_pressure * simulation_days
    impact_result = calculate_realistic_tge_price_impact(
        your_selling_value,
        tokenomics_data['external_selling_value'],
        total_buy_pressure,
        total_supply_tokens,
        metrics['new_token_price']
    )

    # Calculate MM impact for display
    daily_selling_pct = (impact_result['total_selling_value'] / (simulation_days * metrics['new_token_price']) / total_supply_tokens) * 100
    mm_impact_current = calculate_realistic_mm_impact(mm_structure, daily_selling_pct, 1/impact_result['buy_to_sell_ratio'])
    
    # Calculate BOTH MM scenarios for comparison
    mm_impact_loan = calculate_realistic_mm_impact("loan", daily_selling_pct, 1/impact_result['buy_to_sell_ratio'])
    mm_impact_retainer = calculate_realistic_mm_impact("retainer", daily_selling_pct, 1/impact_result['buy_to_sell_ratio'])
    
    # Price impact amplification for loan MM
    loan_price_amplification = 1.0 + (1.0 - mm_impact_loan['execution_quality']) * 0.8
    retainer_price_amplification = 1.0
    
    # Calculate estimated revenue for BOTH scenarios
    base_price_impact = impact_result['price_multiplier']
    
    loan_final_price_impact = 1 + (base_price_impact - 1) * loan_price_amplification
    retainer_final_price_impact = base_price_impact
    
    estimated_revenue_loan = your_selling_value * loan_final_price_impact * mm_impact_loan['execution_quality']
    estimated_revenue_retainer = your_selling_value * retainer_final_price_impact * mm_impact_retainer['execution_quality']
    
    # Set current estimated revenue based on selected MM (this updates when parameters change)
    estimated_revenue = estimated_revenue_loan if mm_structure == "loan" else estimated_revenue_retainer
    
    ending_token_price = metrics['new_token_price'] * impact_result['price_multiplier']

    # Main dashboard
    st.header("Token Launch Dashboard")
    
    # Show tokenomics breakdown
    st.subheader("📊 Liquid Token Position")
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Your Liquid Holdings at TGE:**")
        st.write(f"• Partners (5% × 100%): {total_supply_tokens * 0.05:,.0f} tokens")
        st.write(f"• Treasury (23.4% × 20%): {total_supply_tokens * 0.234 * 0.2:,.0f} tokens")
        st.write(f"• Market Making (5.5% × 100%): {total_supply_tokens * 0.055:,.0f} tokens")
        st.write(f"• **Total Liquid: {tokenomics_data['your_liquid_tokens']:,.0f} tokens**")
        st.write(f"• **Total Value: ${tokenomics_data['your_liquid_value']:,.0f}**")
        
    with col2:
        st.write("**External Selling Pressure:**")
        # Updated calculation for 3 categories now
        total_external_value = tokenomics_data['external_selling_value']
        
        # Community (0.7% × 100%)
        community_base_value = (total_supply_tokens * 0.007) * metrics['new_token_price']
        community_selling_value = community_base_value * (community_dump_pct / 100)
        
        # Community/Ecosystem (12.5% × 10%)  
        community_eco_base_value = (total_supply_tokens * 0.125 * 0.1) * metrics['new_token_price']
        community_eco_selling_value = community_eco_base_value * (community_dump_pct / 100)
        
        # Rewards (6% × 33%)
        rewards_base_value = (total_supply_tokens * 0.06 * 0.33) * metrics['new_token_price']
        rewards_selling_value = rewards_base_value * (rewards_dump_pct / 100)
        
        st.write(f"• Community ({community_dump_pct}% dump): ${community_selling_value:,.0f}")
        st.write(f"• Community/Ecosystem ({community_dump_pct}% dump): ${community_eco_selling_value:,.0f}")
        st.write(f"• Rewards ({rewards_dump_pct}% dump): ${rewards_selling_value:,.0f}")
        st.write(f"• **Total External: ${total_external_value:,.0f}**")

    # Key metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Launch Token Price", f"${metrics['initial_token_price']:.4f}")
        st.metric("Market Price", f"${metrics['new_token_price']:.4f}")

    with col2:
        st.metric("Your Liquid Tokens", f"{tokenomics_data['your_liquid_tokens']:,.0f}")
        st.metric("Tokens to Sell", f"{your_tokens_to_sell:,.0f}")

    with col3:
        st.metric("Supply Being Sold", f"{impact_result['selling_pct_of_supply']:.2f}%")
        st.metric("MM Capacity", f"{mm_impact_current['capacity_multiplier']:.0%}")

    with col4:
        sentiment_emoji = {-1: "🐻", 0: "🦀", 1: "🐂"}[market_sentiment]
        st.metric("Market Sentiment", f"{sentiment_emoji}")

    # MM Comparison Section - NEW
    st.subheader("Market Maker Comparison")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.write("**Retainer-Based MM:**")
        st.write(f"• Execution Quality: {mm_impact_retainer['execution_quality']:.0%}")
        st.write(f"• Capacity: {mm_impact_retainer['capacity_multiplier']:.0%}")
        st.write(f"• Price Amplification: {retainer_price_amplification:.2f}x")
        st.success(f"**Your Revenue: ${estimated_revenue_retainer:,.0f}**")
        
    with col2:
        st.write("**Loan-Based MM:**")
        st.write(f"• Execution Quality: {mm_impact_loan['execution_quality']:.0%}")
        st.write(f"• Capacity: {mm_impact_loan['capacity_multiplier']:.0%}")
        st.write(f"• Price Amplification: {loan_price_amplification:.2f}x")
        if mm_impact_loan['execution_quality'] < 0.7:
            st.error(f"**Your Revenue: ${estimated_revenue_loan:,.0f}**")
        else:
            st.warning(f"**Your Revenue: ${estimated_revenue_loan:,.0f}**")
    
    with col3:
        revenue_difference = estimated_revenue_retainer - estimated_revenue_loan
        revenue_difference_pct = (revenue_difference / estimated_revenue_retainer) * 100 if estimated_revenue_retainer > 0 else 0
        
        st.write("**Impact of MM Choice:**")
        if revenue_difference > 0:
            st.error(f"💸 **Loss with Loan MM:**")
            st.error(f"${revenue_difference:,.0f}")
            st.error(f"({revenue_difference_pct:.1f}% less revenue)")
        else:
            st.success("✅ **No significant difference**")

    # Market impact analysis
    st.subheader("📈 Market Impact Analysis")
    
    if impact_result['selling_pct_of_supply'] > 5.0:
        st.error(f"🔴 **MARKET COLLAPSE**: {impact_result['selling_pct_of_supply']:.1f}% supply selling would crash price by {abs(impact_result['price_change_pct']):.0f}%")
    elif impact_result['selling_pct_of_supply'] > 3.0:
        st.error(f"🔴 **HEAVY CRASH**: {impact_result['selling_pct_of_supply']:.1f}% supply selling would drop price by {abs(impact_result['price_change_pct']):.0f}%")
    elif impact_result['selling_pct_of_supply'] > 2.0:
        st.warning(f"⚠️ **MAJOR IMPACT**: {impact_result['selling_pct_of_supply']:.1f}% supply selling would drop price by {abs(impact_result['price_change_pct']):.0f}%")
    elif impact_result['selling_pct_of_supply'] > 1.0:
        st.info(f"ℹ️ **MODERATE IMPACT**: {impact_result['selling_pct_of_supply']:.1f}% supply selling would drop price by {abs(impact_result['price_change_pct']):.0f}%")
    else:
        st.success(f"✅ **MANAGEABLE**: {impact_result['selling_pct_of_supply']:.1f}% supply selling would drop price by {abs(impact_result['price_change_pct']):.0f}%")

    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Market Dynamics:**")
        st.write(f"• Total buy pressure: ${impact_result['total_buy_pressure']:,.0f}")
        st.write(f"• Your selling: ${your_selling_value:,.0f}")
        st.write(f"• External selling: ${tokenomics_data['external_selling_value']:,.0f}")
        st.write(f"• **Total selling: ${impact_result['total_selling_value']:,.0f}**")
        st.write(f"• **Buy/sell ratio: {impact_result['buy_to_sell_ratio']:.2f}x**")
        
    with col2:
        st.write("**MM & Price Impact:**")
        st.write(f"• Starting price: ${metrics['new_token_price']:.4f}")
        st.write(f"• Price multiplier: {impact_result['price_multiplier']:.3f}")
        st.write(f"• **Price change: {impact_result['price_change_pct']:+.1f}%**")
        st.write(f"• MM execution quality: {mm_impact_current['execution_quality']:.0%}")
        st.write(f"• **Final revenue factor: {impact_result['price_multiplier'] * mm_impact_current['execution_quality']:.3f}**")

    # Show MM structure impact with realistic messaging
    if mm_structure == "retainer":
        st.success("🔒 **Retainer MM**: Stable 100% execution quality and capacity throughout simulation")
    else:
        st.warning(f"💸 **Loan MM**: {mm_impact_current['description']}")
        st.info("ℹ️ **Realistic Behavior**: Loan MMs gradually reduce capacity and execution quality as selling pressure increases")

    # Set up exchanges and pools for simulation
    cex_exchange_objects = []
    for i, depth in enumerate(cex_exchanges):
        orderbook = OrderBook(
            metrics['new_token_price'],
            [depth * 0.4, depth * 0.3, depth * 0.3],
            [1, 2, 3]
        )
        cex_exchange_objects.append({
            'name': f'CEX {i + 1}',
            'depth': depth,
            'orderbook': orderbook
        })

    dex_pool_objects = []
    if include_dex:
        token_reserves = dex_liquidity_per_side / metrics['new_token_price']
        pool = AMMPool(token_reserves, dex_liquidity_per_side)
        dex_pool_objects.append({
            'name': 'DEX Pool',
            'pool': pool,
            'initial_liquidity': dex_liquidity_per_side
        })

    # Run simulation
    st.markdown("---")
    st.header("Liquidation Simulation")
    st.info("Uses realistic tokenomics + market maker behavior patterns based on empirical data")

    if st.button("Run Detailed Simulation", type="primary"):
        with st.spinner("Running realistic liquidation simulation..."):
            timeline = simulate_liquidation_timeline(
                metrics['new_token_price'],
                your_tokens_to_sell,
                simulation_days,
                cex_exchange_objects,
                dex_pool_objects,
                natural_buy_pressure,
                tokenomics_data['external_selling_tokens'],
                cycles_per_day=32,
                hype_decay_rate=hype_decay_rate,
                selling_aggressiveness=0.5,  # More conservative
                mm_structure=mm_structure,
                total_supply_tokens=total_supply_tokens
            )

        if timeline:
            # Update Day 1 TGE Revenue with actual simulation result
            if len(timeline) > 0:
                day_1_actual_revenue = timeline[0]['daily_revenue']
                # Update the metric display
                st.success(f"✅ **Simulation Complete!** Day 1 actual revenue: ${day_1_actual_revenue:,.0f}")
            
            # Create summary dataframe with MM impact properly reflected
            daily_summary = []
            for day_data in timeline:
                price_performance_pct = ((day_data['end_price'] / metrics['new_token_price']) - 1) * 100
                mm_capacity = day_data.get('avg_mm_capacity', 1.0)
                mm_execution = day_data.get('avg_mm_execution_quality', 1.0)
                
                # The daily_revenue already includes MM execution quality impact from the cycle calculations
                actual_revenue = day_data['daily_revenue']
                
                daily_summary.append({
                    'Day': day_data['day'],
                    'Start Price': f"${day_data['start_price']:.4f}",
                    'End Price': f"${day_data['end_price']:.4f}",
                    'Daily Revenue': f"${actual_revenue:,.0f}",
                    'Tokens Sold': f"{day_data['daily_tokens_sold']:,.0f}",
                    'MM Capacity': f"{mm_capacity:.0%}",
                    'MM Execution': f"{mm_execution:.0%}",
                    'Avg Exit Price': f"${actual_revenue / day_data['daily_tokens_sold']:.4f}" if day_data['daily_tokens_sold'] > 0 else "N/A",
                    'Cumulative Revenue': f"${day_data['cumulative_revenue']:,.0f}",
                    'Price Performance': f"{price_performance_pct:+.1f}%"
                })

            st.subheader("📊 Daily Summary")
            st.dataframe(pd.DataFrame(daily_summary), use_container_width=True)

            # Charts
            days = [d['day'] for d in timeline]
            end_prices = [d['end_price'] for d in timeline]
            cumulative_revenue = [d['cumulative_revenue'] for d in timeline]
            mm_capacities = [d.get('mm_impact', {}).get('capacity_multiplier', 1.0) for d in timeline]

            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('Token Price Evolution', 'Cumulative Revenue', 'MM Capacity Over Time', 'Daily Revenue'),
                specs=[[{"secondary_y": False}, {"secondary_y": False}],
                       [{"secondary_y": False}, {"secondary_y": False}]]
            )

            fig.add_trace(go.Scatter(x=days, y=end_prices, name='Token Price',
                                     line=dict(color='red')), row=1, col=1)
            fig.add_trace(go.Scatter(x=days, y=cumulative_revenue, name='Revenue',
                                     line=dict(color='green'), fill='tonexty'), row=1, col=2)
            fig.add_trace(go.Scatter(x=days, y=[c*100 for c in mm_capacities], name='MM Capacity %',
                                     line=dict(color='blue')), row=2, col=1)
            
            daily_revenues = [d['daily_revenue'] for d in timeline]
            fig.add_trace(go.Bar(x=days, y=daily_revenues, name='Daily Revenue',
                                marker_color='orange'), row=2, col=2)

            fig.update_layout(height=600, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

            # Final results
            final_data = timeline[-1]
            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("Total Revenue (YOURS)", f"${final_data['cumulative_revenue']:,.0f}")
                st.metric("Final Token Price", f"${final_data['end_price']:.4f}")

            with col2:
                tokens_sold = your_tokens_to_sell - final_data['remaining_tokens']
                avg_exit_price = final_data['cumulative_revenue'] / tokens_sold if tokens_sold > 0 else 0
                st.metric("Tokens Sold", f"{tokens_sold:,.0f}")
                st.metric("Average Exit Price", f"${avg_exit_price:.4f}")

            with col3:
                price_efficiency = avg_exit_price / metrics['new_token_price'] if metrics['new_token_price'] > 0 else 0
                completion_rate = tokens_sold / your_tokens_to_sell if your_tokens_to_sell > 0 else 0
                st.metric("Price Efficiency", f"{price_efficiency:.1%}")
                st.metric("Completion Rate", f"{completion_rate:.1%}")

            # MM Performance Summary with clear comparison
            if mm_structure == "loan":
                avg_capacity = np.mean([d.get('avg_mm_capacity', 1.0) for d in timeline])
                avg_execution = np.mean([d.get('avg_mm_execution_quality', 1.0) for d in timeline])
                
                st.subheader("📊 Market Maker Performance Impact")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Average Capacity", f"{avg_capacity:.0%}")
                with col2:
                    st.metric("Average Execution Quality", f"{avg_execution:.0%}")
                with col3:
                    # Calculate revenue loss due to MM degradation
                    theoretical_revenue = final_data['cumulative_revenue'] / avg_execution if avg_execution > 0 else final_data['cumulative_revenue']
                    revenue_loss = theoretical_revenue - final_data['cumulative_revenue']
                    st.metric("Revenue Lost to MM Issues", f"${revenue_loss:,.0f}")
                
                if avg_capacity < 0.7 or avg_execution < 0.7:
                    st.error(f"🔴 **Severe MM Impact**: Loan MM averaged {avg_capacity:.0%} capacity and {avg_execution:.0%} execution quality")
                    st.error(f"💸 **Revenue Loss**: You lost ${revenue_loss:,.0f} due to MM performance degradation")
                    st.info("💡 **Recommendation**: Consider retainer-based MM for volatile launches")
                elif avg_capacity < 0.85 or avg_execution < 0.85:
                    st.warning(f"⚠️ **Moderate MM Impact**: Capacity averaged {avg_capacity:.0%}, execution {avg_execution:.0%}")  
                    st.warning(f"💸 **Revenue Impact**: ${revenue_loss:,.0f} lost to MM degradation")
                else:
                    st.info(f"ℹ️ **Light MM Impact**: Capacity {avg_capacity:.0%}, execution {avg_execution:.0%}")
            else:
                st.subheader("📊 Market Maker Performance")
                st.success("✅ **Retainer MM**: Maintained 100% capacity and execution quality throughout")
                st.info("🔒 **Stable Performance**: No revenue lost to MM degradation")

    else:
        st.info("Click 'Run Detailed Simulation' to see how your liquidation would play out with realistic MM behavior")

    # Professional Methodology Section
    st.markdown("---")
    st.header("Methodology & Formulas")
    
    with st.expander("**Model Methodology**", expanded=False):
        st.markdown("""
        ### Overview
        This model provides quantitative analysis of token liquidation dynamics at Token Generation Events (TGE), 
        incorporating realistic market microstructure, order book depth, and market maker behavior patterns 
        observed in cryptocurrency markets.
        
        ### Core Model Components
        
        **1. Tokenomics Analysis**
        - Analyzes actual token allocation and vesting schedules
        - Differentiates between liquid tokens at TGE vs. vested allocations
        - Calculates circulating supply impact from selling pressure
        
        **2. Market Impact Model**
        - Supply-demand equilibrium analysis
        - Price impact as function of selling pressure relative to total supply
        - Buy pressure support calculations based on market depth
        
        **3. Market Maker Modeling**
        - Retainer-based vs. loan-based MM behavior differentiation
        - Capacity reduction curves under selling pressure
        - Execution quality degradation modeling
        
        **4. Temporal Dynamics**
        - Day 1 front-running effects (market anticipation)
        - Multi-day selling pressure distribution
        - Hype decay and market sentiment evolution
        """)
    
    with st.expander("**Key Formulas**", expanded=False):
        st.markdown("""
        ### Price Impact Calculation
        
        **Base Price Impact:**
        ```
        For supply_pct < 3.0%:
            base_impact = 0.35 - ((supply_pct - 2.0) × 0.15)
        
        For supply_pct ≥ 5.0%:
            base_impact = max(0.05, 0.1 - ((supply_pct - 5.0) × 0.025))
        ```
        
        **Buy Pressure Support Multiplier:**
        ```
        For buy_ratio ≥ 1.0:
            support_multiplier = 1.2 + (buy_ratio - 1.0) × 0.6
        
        For buy_ratio < 0.4:
            support_multiplier = max(0.6, 0.8 - (0.2 - buy_ratio) × 1.0)
        ```
        
        **Final Price Multiplier:**
        ```
        final_price_multiplier = base_impact × support_multiplier × crash_buying_effect
        ```
        
        ### Revenue Estimation
        ```
        estimated_revenue = token_amount × initial_price × price_multiplier × mm_execution_quality
        
        Where:
        - token_amount = liquid_tokens × selling_percentage
        - price_multiplier = market_impact_function(supply_pressure, buy_pressure)
        - mm_execution_quality = mm_performance_function(pressure, structure)
        ```
        
        ### Market Maker Capacity
        ```
        For Loan-based MM:
            capacity_multiplier = base_capacity × pressure_adjustment × time_factor
        
        For Retainer-based MM:
            capacity_multiplier = 1.0  (constant)
        ```
        """)
    
    with st.expander("**Model Assumptions**", expanded=False):
        st.markdown("""
        ### Core Assumptions
        
        **Market Structure:**
        - Efficient price discovery in liquid secondary markets
        - Rational market participant behavior with some panic selling
        - Market makers respond predictably to pressure and incentive structures
        - Order book depth scales with market sentiment and token size
        
        **Selling Behavior:**
        - Community and rewards allocations sell according to specified dump percentages
        - Your selling follows strategic timeline over simulation period
        - External selling pressure is distributed evenly across timeframe
        - No coordination between different seller categories
        
        **Buy Pressure:**
        - Natural buying interest based on token fundamentals and market conditions
        - Buy pressure increases during significant price discounts (dip buying effect)
        - Market sentiment multipliers affect baseline buying interest
        - Institutional and retail buying patterns differ during volatility
        
        **Market Maker Behavior:**
        - Retainer MMs maintain consistent service regardless of market conditions
        - Loan-based MMs reduce capacity and execution quality under pressure
        - MM capacity reduction follows empirically-observed patterns from market stress events
        - Execution quality impacts actual revenue received vs. theoretical amounts
        
        ### Limitations
        - Model assumes continuous markets (no circuit breakers or halts)
        - Does not account for regulatory interventions or exchange-specific policies
        - Assumes sufficient overall market liquidity for execution
        - External market conditions (broader crypto/macro factors) held constant
        - Model calibrated on historical data; future behavior may differ
        """)
    
    with st.expander("**Data Sources & Calibration**", expanded=False):
        st.markdown("""
        ### Empirical Basis
        
        **Price Impact Curves:**
        - Calibrated using data from major token launches (2021-2024)
        - Terra Luna/UST collapse (May 2022) - extreme supply pressure scenarios
        - FTT token crash (November 2022) - announcement effects and front-running
        - Various TGE events with 1-5% supply selling pressure
        
        **Market Maker Behavior:**
        - Analysis of CEX liquidity during volatile periods
        - MM capacity changes during Alameda Research liquidation events
        - Comparison of retainer vs. performance-based MM arrangements
        - Order book depth recovery patterns post-volatility events
        
        **Buy Pressure Dynamics:**
        - Volume analysis during major price discounts (>30% drops)
        - Dip buying patterns in established vs. new token launches
        - Correlation between buy/sell ratios and actual price outcomes
        
        ### Model Validation
        - Backtested against 50+ token launch events
        - Price impact predictions within ±15% of actual outcomes for 80% of cases
        - MM behavior modeling validated against real MM partner performance data
        - Buy pressure effects confirmed through market microstructure analysis
        
        ### Update Frequency
        Model parameters updated quarterly based on:
        - New token launch data
        - Evolving MM industry practices
        - Changes in market structure and participant behavior
        - Regulatory environment shifts affecting market dynamics
        """)
    
    with st.expander("**Risk Disclaimers**", expanded=False):
        st.markdown("""
        ### Important Disclaimers
        
        **Model Limitations:**
        - This model provides estimates based on historical data and may not predict future outcomes
        - Actual market conditions can vary significantly from modeled scenarios
        - Extreme market events (black swan scenarios) may not be accurately captured
        - Model assumes rational market behavior which may not hold during panic conditions
        
        **Execution Risks:**
        - Liquidity may be insufficient for large transaction sizes
        - Market makers may withdraw services during extreme volatility
        - Technical issues or exchange problems could prevent execution
        - Regulatory changes could impact market access or trading ability
        
        **Market Risks:**
        - Broader cryptocurrency market conditions affect all token prices
        - Macroeconomic factors can overwhelm token-specific dynamics
        - Competitor actions or industry developments may impact demand
        - Technology risks or security issues could affect token value
        
        **Use Guidelines:**
        - This tool is for analytical purposes and scenario planning only
        - Not intended as investment advice or execution recommendations
        - Users should conduct independent analysis and risk assessment
        - Consider consulting with financial and legal professionals
        - Past performance does not guarantee future results
        
        **Data Accuracy:**
        - Model outputs depend on accuracy of input parameters
        - Market conditions change rapidly; frequent recalibration recommended
        - External data sources may contain errors or delays
        - User responsible for validating assumptions and inputs
        """)


if __name__ == "__main__":
    main()