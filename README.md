# RAG_QA for 海外旅行不便險條款

本專案實作一個 **檢索增強生成 (RAG)** 系統，針對《海外旅行不便險條款》PDF 提供自動化問答 
使用者可輸入自然語言問題，系統會檢索相關條文並生成結構化回答

---

## 📂 專案架構

- `rag_with_ollama.py`  
  主程式，負責 PDF 解析、檢索、主題路由、答案生成
- `request.txt`  
  所需套件清單，可用於安裝環境。
- `海外旅行不便險條款.pdf`  
  測試用保險條款文件（需自行提供）。

---

## ⚙️ 安裝與環境

1. 建立虛擬環境：
   ```bash
   python3 -m venv .env
   source .env/bin/activate
2. 安裝套件
   ```bash
   pip install -r request.txt
3. 安裝並啟動ollama，確保可以呼叫本地模型，預設模型為 `qwen2:7b`，可透過環境變數調整
   ```bash
   export OLLAMA_MODEL="qwen2:7b"

---


## 📂 問答模式

```bash
python rag_travel.py --pdf ./海外旅行不便險條款.pdf --q "什麼情況下可以申請旅遊延誤賠償？"
```



輸出內容包含Route, Answer, Sources

