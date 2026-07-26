
from __future__ import annotations

from pathlib import Path
import time

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from src.data_utils import forecast_returns, load_price_data
from src.portfolio import (
    build_selection_qubo,
    equal_weight_solution,
    evaluate_realized,
    exact_select,
    optimize_weights,
)
from src.qaoa_numpy import solve_qaoa


st.set_page_config(
    page_title="AI–Quantum Portfolio Intelligence",
    page_icon="🧠",
    layout="wide",
)

PROJECT_ROOT = Path(__file__).parent
DEMO_FILE = PROJECT_ROOT / "data" / "demo_prices.csv"

st.title("AI–Quantum Portfolio Intelligence Platform")
st.caption(
    "PoC: AI dự báo đầu vào → QAOA chọn tài sản → "
    "tối ưu cổ điển phân bổ tỷ trọng → dashboard hỗ trợ quyết định."
)
st.warning(
    "Sản phẩm phục vụ nghiên cứu và minh họa kỹ thuật, không phải khuyến nghị đầu tư "
    "và không tự động đặt lệnh."
)

with st.sidebar:
    st.header("Cấu hình")
    data_mode = st.radio(
        "Nguồn dữ liệu",
        ["Dữ liệu demo mô phỏng", "Tải CSV của nhóm"],
    )
    uploaded = None
    if data_mode == "Tải CSV của nhóm":
        uploaded = st.file_uploader("Tải file CSV", type=["csv"])

    st.divider()
    cardinality = st.slider("Số tài sản cần chọn (K)", 2, 6, 4)
    risk_aversion = st.slider("Mức ngại rủi ro λ", 0.1, 10.0, 3.0, 0.1)
    max_weight = st.slider("Tỷ trọng tối đa mỗi tài sản", 0.20, 0.80, 0.50, 0.05)
    lookback = st.selectbox("Cửa sổ ước lượng", [126, 252, 504], index=1)
    penalty = st.slider("Hệ số phạt ràng buộc QUBO", 0.5, 10.0, 4.0, 0.5)
    run = st.button("Chạy mô hình", type="primary", use_container_width=True)


def get_prices() -> pd.DataFrame:
    if data_mode == "Dữ liệu demo mô phỏng":
        st.info(
            "Ứng dụng đang dùng dữ liệu mô phỏng để bảo đảm demo chạy ổn định. "
            "Trước khi nộp chính thức, nhóm nên thay bằng dữ liệu lịch sử hợp pháp."
        )
        return load_price_data(DEMO_FILE)
    if uploaded is None:
        st.stop()
    return load_price_data(uploaded)


try:
    prices = get_prices()
except Exception as exc:
    st.error(f"Không thể đọc dữ liệu: {exc}")
    st.stop()

cardinality = min(cardinality, len(prices.columns) - 1)

tab_overview, tab_data, tab_model, tab_results, tab_method = st.tabs(
    ["Tổng quan", "Dữ liệu", "AI Forecast", "Tối ưu danh mục", "Phương pháp"]
)

with tab_overview:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Số tài sản", len(prices.columns))
    c2.metric("Số phiên", len(prices))
    c3.metric("Từ ngày", prices.index.min().strftime("%d/%m/%Y"))
    c4.metric("Đến ngày", prices.index.max().strftime("%d/%m/%Y"))

    normalized = prices / prices.iloc[0] * 100
    fig = px.line(
        normalized,
        x=normalized.index,
        y=normalized.columns,
        labels={"value": "Chỉ số giá (gốc = 100)", "Date": "Ngày", "variable": "Mã"},
        title="Diễn biến giá chuẩn hóa",
    )
    st.plotly_chart(fig, use_container_width=True)

with tab_data:
    st.subheader("Mẫu dữ liệu đầu vào")
    st.dataframe(prices.tail(20), use_container_width=True)
    st.download_button(
        "Tải mẫu CSV",
        data=prices.reset_index().to_csv(index=False).encode("utf-8"),
        file_name="price_data_template.csv",
        mime="text/csv",
    )

if run:
    with st.spinner("Đang huấn luyện AI, xây QUBO và mô phỏng QAOA..."):
        started = time.perf_counter()
        forecast = forecast_returns(prices, lookback=lookback)

        assets = list(prices.columns)
        Q = build_selection_qubo(
            forecast.expected_returns,
            forecast.covariance,
            cardinality,
            risk_aversion,
            penalty,
        )

        classical_bits = exact_select(
            forecast.expected_returns,
            forecast.covariance,
            cardinality,
            risk_aversion,
        )
        qaoa_result = solve_qaoa(Q, cardinality=cardinality)

        classical_assets = [a for a, b in zip(assets, classical_bits) if b == 1]
        qaoa_assets = [a for a, b in zip(assets, qaoa_result.bitstring) if b == 1]

        classical_solution = optimize_weights(
            classical_assets,
            forecast.expected_returns,
            forecast.covariance,
            risk_aversion,
            max_weight,
        )
        qaoa_solution = optimize_weights(
            qaoa_assets,
            forecast.expected_returns,
            forecast.covariance,
            risk_aversion,
            max_weight,
        )
        equal_solution = equal_weight_solution(
            assets,
            forecast.expected_returns,
            forecast.covariance,
            risk_aversion,
        )

        elapsed = time.perf_counter() - started

    st.session_state["results"] = {
        "forecast": forecast,
        "Q": Q,
        "classical_solution": classical_solution,
        "qaoa_solution": qaoa_solution,
        "equal_solution": equal_solution,
        "qaoa_result": qaoa_result,
        "elapsed": elapsed,
    }

results = st.session_state.get("results")

with tab_model:
    if results is None:
        st.info("Nhấn **Chạy mô hình** để xem kết quả dự báo.")
    else:
        score = results["forecast"].model_scores.copy()
        score["Predicted_Annual_Return"] = score["Predicted_Annual_Return"].map(
            lambda x: f"{x:.2%}"
        )
        score["MAE_daily"] = score["MAE_daily"].map(
            lambda x: "" if pd.isna(x) else f"{x:.5f}"
        )
        score["Directional_Accuracy"] = score["Directional_Accuracy"].map(
            lambda x: "" if pd.isna(x) else f"{x:.1%}"
        )
        st.subheader("Kết quả dự báo")
        st.dataframe(score, use_container_width=True)

with tab_results:
    if results is None:
        st.info("Nhấn **Chạy mô hình** để tối ưu danh mục.")
    else:
        qaoa_solution = results["qaoa_solution"]
        classical_solution = results["classical_solution"]
        equal_solution = results["equal_solution"]
        qaoa_result = results["qaoa_result"]

        st.success(f"Hoàn thành trong {results['elapsed']:.2f} giây.")

        metric_rows = []
        for name, sol in [
            ("Equal-weight", equal_solution),
            ("Classical exact + weight optimization", classical_solution),
            ("QAOA + weight optimization", qaoa_solution),
        ]:
            realized = evaluate_realized(
                results["forecast"].returns,
                sol.weights,
                evaluation_days=min(60, len(results["forecast"].returns)),
            )
            metric_rows.append(
                {
                    "Phương pháp": name,
                    "Expected Return": sol.expected_return,
                    "Volatility": sol.volatility,
                    "Expected Sharpe": sol.sharpe,
                    "Realized Return": realized["Realized Return"],
                    "Realized Sharpe": realized["Realized Sharpe"],
                    "Maximum Drawdown": realized["Maximum Drawdown"],
                    "Objective": sol.objective,
                }
            )

        comparison = pd.DataFrame(metric_rows).set_index("Phương pháp")
        st.subheader("So sánh phương pháp")
        st.dataframe(
            comparison.style.format(
                {
                    "Expected Return": "{:.2%}",
                    "Volatility": "{:.2%}",
                    "Expected Sharpe": "{:.3f}",
                    "Realized Return": "{:.2%}",
                    "Realized Sharpe": "{:.3f}",
                    "Maximum Drawdown": "{:.2%}",
                    "Objective": "{:.5f}",
                }
            ),
            use_container_width=True,
        )

        left, right = st.columns(2)
        with left:
            st.subheader("Danh mục QAOA Hybrid")
            weight_df = (
                qaoa_solution.weights.sort_values(ascending=False)
                .rename("Tỷ trọng")
                .reset_index(names="Mã")
            )
            fig_w = px.bar(
                weight_df,
                x="Mã",
                y="Tỷ trọng",
                text_auto=".1%",
                title="Tỷ trọng phân bổ cuối cùng",
            )
            st.plotly_chart(fig_w, use_container_width=True)
            st.dataframe(
                weight_df.style.format({"Tỷ trọng": "{:.2%}"}),
                use_container_width=True,
                hide_index=True,
            )

        with right:
            st.subheader("Thông tin QAOA p=1")
            a, b, c = st.columns(3)
            a.metric("γ", f"{qaoa_result.gamma:.3f}")
            b.metric("β", f"{qaoa_result.beta:.3f}")
            c.metric("Xác suất nghiệm", f"{qaoa_result.probability:.2%}")
            st.caption(
                "QAOA được mô phỏng bằng statevector NumPy ở quy mô nhỏ. "
                "Kết quả không dùng để khẳng định quantum speedup."
            )
            st.dataframe(pd.DataFrame(qaoa_result.top_states), use_container_width=True)

        st.subheader("Giải thích quyết định")
        explanation = pd.DataFrame(
            {
                "Predicted Annual Return": results["forecast"].expected_returns,
                "Risk Proxy": np.sqrt(np.diag(results["forecast"].covariance)),
                "QAOA Selected": [
                    a in qaoa_solution.selected
                    for a in results["forecast"].expected_returns.index
                ],
                "Final Weight": [
                    float(qaoa_solution.weights.get(a, 0.0))
                    for a in results["forecast"].expected_returns.index
                ],
            }
        )
        st.dataframe(
            explanation.style.format(
                {
                    "Predicted Annual Return": "{:.2%}",
                    "Risk Proxy": "{:.2%}",
                    "Final Weight": "{:.2%}",
                }
            ),
            use_container_width=True,
        )

with tab_method:
    st.markdown(
        """
### Kiến trúc PoC

1. **Dữ liệu:** giá lịch sử theo ngày của 8–12 cổ phiếu.
2. **AI Prediction:** Gradient Boosting ước lượng lợi nhuận kỳ vọng.
3. **QUBO:** mã hóa mục tiêu lợi nhuận–rủi ro và ràng buộc chọn đúng K tài sản.
4. **QAOA p=1:** mô phỏng statevector để chọn tập tài sản.
5. **Classical optimization:** tối ưu tỷ trọng trong tập tài sản QAOA đã chọn.
6. **Benchmark:** equal-weight và exact classical selection.
7. **Dashboard:** so sánh hiệu quả, rủi ro và giải thích lựa chọn.

### Giới hạn

- Không giao dịch tiền thật.
- Không dự báo thời gian thực từng giây.
- Không chạy hàng trăm tài sản trên simulator.
- Không khẳng định lợi thế tốc độ lượng tử.
- Không thay thế chuyên gia quản lý danh mục.
"""
    )
