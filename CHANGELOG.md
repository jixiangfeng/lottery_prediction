# Changelog
All notable changes to this project will be documented in this file.
## [Unreleased] - 2026-07-16
### Added
- 新增福彩3D、排列三、排列五严格逐期前推回测、`uniform_random` 基线、JSON/Markdown 报告与 `make digit-walk-forward` 入口。
- 数字彩统计新增 20/50/100/300 多窗口位置频率、贝叶斯平滑概率和公开权重配置。
- 数字彩统计新增所有位置对联合频率/平滑概率，以及形态、和值、跨度多窗口平滑概率。
- 新增直选/组选分离候选 API；三位彩组选按无序 key 聚合过滤空间归一化模型质量，排列五只返回直选。
- 严格前推新增多随机基线分布、当前策略百分位、候选复合模型分/目标排名和可选嵌套调参证据。

### Changed
- 数字彩候选从随机抽样升级为三位 1000 种、五位 100000 种全空间确定性评分与高分池多样性选择。
- 遗漏分改为对数压缩并封顶；三位彩允许跨度 1 的组三，排列五增加少量“三一一/三二”防守形态。
- 数字彩主目标从位置覆盖调整为启发式复合模型排序；多样性只在 `score_floor` 高质量池内作为约束和同分破局，该评分不是实际开奖概率。
- 排列三与排列五前三位使用同一共享评分函数，旧日报 JSON 字段保持兼容并新增 `directCandidates`/`groupCandidates`。

### Fixed
- 严格前推训练集截止到目标期前一期，避免固定候选历史回放中的未来数据泄漏。
- 统一统计直选、随机直选、统计组选、随机组选的严格形态预算，并保证组选返回数量等于请求数量。
- 嵌套调参统一使用最终直选/组选候选判定命中，跨配置比较仅使用命中数与过滤空间归一化排名分位。
- 将默认 `score_floor` 收紧至 `2.0`，并确保多样性选择不低于纯复合模型分 TopK 的最低分。

## [1.4.0] - 2025-10-14 🎲 Copula 采样与图嵌入融合
### Added
- 新增 `src/analysis/copula_sampler.py` 与 `generate_copula_candidates`，在高级模式中提供 Copula 多样性采样。
- 新增 `scripts/train_graph_embeddings.py`（PyTorch Node2Vec），支持 `--device auto`，输出嵌入缓存。
- 新增 `src/analysis/mutual_information.py`，在高级评分阶段引入互信息多样性惩罚。
- 新增测试 `tests/test_copula_sampler.py`、`tests/test_mutual_information.py`，并扩展 `test_feature_enhancer.py`。
### Changed
- `feature_enhancer` 融合 `graph_embedding_scores`，输出调试字段并提供缓存清理函数。
- `advanced_number_generation` 集成 Copula 候选与互信息扣分；Makefile 增加 `train-graph` 目标。
- `config/config.yaml` 补充 `analysis.copula`、`analysis.graph_embedding` 默认配置，CLI 允许覆盖。
### Documentation
- 更新 README、`docs/kl8_algorithm_extension_report.md`、`docs/decision_record.md`、`ASSUMPTIONS.md`、`docs/api.md`、`agent_report.md`，补充 Copula/图嵌入说明。

## [1.3.0] - 2025-10-13 馃 Dirichlet 骞虫粦涓庤鍒欑瓫閫?
### Added
- `compute_enhanced_scores` 鏂板 Dirichlet-Multinomial 鍚庨獙閫氶亾锛屾潈閲嶅彲閫氳繃 `config.analysis.dirichlet` 閰嶇疆銆?
- 鏂板 `src/analysis/rule_miner.py`锛屾敮鎸?FP-Growth 棰戠箒椤归泦缂撳瓨涓?`--rule_filter` 杞?纭ā寮忋€?
- 鏂板 `tests/test_rule_miner.py` 瑕嗙洊杞?纭ā寮忚瘎浼伴€昏緫銆?
### Changed
- `kl8_analysis.py` 涓?`kl8_analysis_plus.py` 铻嶅悎 Dirichlet 寰楀垎骞舵帴鍏ヨ鍒欐儵缃氾紝鏆撮湶 `--rule_support`銆乣--rule_confidence` 绛夊弬鏁般€?
### Documentation
- 鏇存柊 README銆乨ocs/api.md銆乨ocs/decision_record.md銆丄SSUMPTIONS.md锛岃ˉ鍏?Dirichlet 閰嶇疆涓庤鍒欑瓫閫夌ず渚嬨€?



## [1.2.3] - 2025-10-12 馃洜锔?鐩稿瀵煎叆闂鍏ㄩ潰淇
### Fixed
- **鐩稿瀵煎叆閿欒淇**锛氫慨澶嶆墍鏈夊彲鐙珛杩愯鑴氭湰鐨勭浉瀵瑰鍏ラ棶棰橈紝纭繚鑴氭湰鍙互鐩存帴杩愯
  - `kl8_analysis.py`锛氫慨澶?涓猘nalysis_metrics瀵煎叆鍜?涓猻hared_utils瀵煎叆
  - `kl8_analysis_plus.py`锛氫慨澶?涓猘nalysis_metrics瀵煎叆鍜?涓猻hared_utils瀵煎叆
  - `analysis_metrics.py`锛氫慨澶峴hared_utils瀵煎叆
- **瀵煎叆绛栫暐缁熶竴**锛氭墍鏈変慨澶嶉兘閲囩敤try-except妯″紡锛屼紭鍏堢浉瀵瑰鍏ワ紝澶辫触鏃跺洖閫€鍒扮粷瀵瑰鍏?
- **鑴氭湰鐩存帴杩愯鏀寔**锛氱‘淇濇墍鏈夊垎鏋愯剼鏈兘鍙互浣滀负鐙珛绋嬪簭杩愯锛屾棤闇€鍖呯粨鏋?

### Validated
- 鉁?`kl8_analysis.py` - 鍙甯哥洿鎺ヨ繍琛岋紝advanced_mode 2 鍜?feature_mode hybrid 宸ヤ綔姝ｅ父
- 鉁?`kl8_analysis_plus.py` - 瀵煎叆閿欒宸蹭慨澶嶏紝鍙甯告樉绀哄府鍔╀俊鎭?
- 鉁?`kl8_cash.py`, `kl8_cash_plus.py`, `kl8_running.py` - 纭鏃犲鍏ラ棶棰?
- 鉁?鎵€鏈夋ā鍧楅兘鍙互姝ｅ父浣滀负鍖呭鍏ヤ娇鐢?

### Technical Details
- 淇ImportError: "attempted relative import with no known parent package"
- 淇濇寔鍚戝悗鍏煎鎬э紝妯″潡瀵煎叆鍜岃剼鏈洿鎺ヨ繍琛岄兘鏀寔
- 閲囩敤缁熶竴鐨勯敊璇鐞嗘ā寮忥紝纭繚浠ｇ爜涓€鑷存€?

## [1.2.2] - 2025-01-25 馃殌 ThreadPoolExecutor鍗囩骇涓庢祴璇曞畬鍠?
### Changed
- **ThreadPoolExecutor鍗囩骇**锛氬皢 `kl8_running.py` 浠庡熀纭€ `threading.Thread` 鍗囩骇涓?`concurrent.futures.ThreadPoolExecutor`
- **鏀硅繘鐨勮祫婧愭帶鍒?*锛氭洿濂界殑绾跨▼姹犵鐞嗗拰寮傚父澶勭悊锛屼娇鐢?`as_completed()` 瀹炵幇鏇翠紭闆呯殑浠诲姟杩涘害璺熻釜
- **澧炲己鐨勯敊璇鐞?*锛氭坊鍔犲け璐ヤ换鍔＄粺璁″拰璇︾粏閿欒鎶ュ憡

### Added
- **瀹屾暣娴嬭瘯瑕嗙洊**锛氫负鎵€鏈夊叡浜伐鍏锋ā鍧楁坊鍔犲崟鍏冩祴璇曪紙`test_shared_utils.py`, `test_shared_download.py`, `test_kl8_running.py`锛?
- **ThreadPoolExecutor娴嬭瘯**锛氶獙璇佸苟鍙戞墽琛屻€佷换鍔″垎鍙戝拰閿欒澶勭悊鐨勪笓闂ㄦ祴璇?
- **鎬ц兘鏂囨。鏇存柊**锛氬湪 README 鍜?docs 涓坊鍔犺缁嗙殑 max_workers 閰嶇疆鎸囧崡鍜屾€ц兘琛?

### Fixed
- **浠ｇ爜璐ㄩ噺鎻愬崌**锛氭秷闄ゆ柊鍏变韩妯″潡鐨刲int璀﹀憡锛岀‘淇濇墍鏈夋柊浠ｇ爜杈惧埌涓ユ牸璐ㄩ噺鏍囧噯
- **娴嬭瘯绋冲畾鎬?*锛氫慨澶嶅嚱鏁扮鍚嶄笉鍖归厤闂锛岀‘淇濇祴璇曚笌瀹為檯瀹炵幇涓€鑷?

### Documentation
- 娣诲姞 max_workers 鎬ц兘浼樺寲琛ㄦ牸鍜岀郴缁熼厤缃缓璁?
- 鏇存柊杩愯鎸囧崡锛屽寘鍚祫婧愮洃鎺у拰娓愯繘璋冧紭绛栫暐

## [1.2.1] - 2025-10-11 馃敡 Running鑴氭湰涓庤嚜閫傚簲闃堝€间紭鍖?
### Fixed
- **kl8_running.py 澶氱嚎绋嬩笅杞介棶棰?*锛氱粺涓€鍦ㄤ富绾跨▼涓嬭浇鏁版嵁锛岄伩鍏嶅苟鍙戝啿绐?
- **鐩綍鍒涘缓绔炰簤鏉′欢**锛氫慨澶?`os.makedirs` 浣跨敤 `exist_ok=True` 鍙傛暟
- **鏂囦欢璺緞闂**锛氫娇鐢ㄧ粷瀵硅矾寰勫畾浣?plus 鐗堟湰鑴氭湰鏂囦欢
- **鑷€傚簲闃堝€肩簿搴?*锛氬畬鍏ㄧЩ闄?`shifting_rate` 渚濊禆锛屼娇鐢ㄩ€掑噺姝ラ暱+鎸囨暟骞虫粦閫昏緫

### Changed
- `kl8_running.py` 鏂板 `--download` 鍙傛暟鎺у埗鏁版嵁涓嬭浇琛屼负
- `adaptive_threshold_update` 鍑芥暟浣跨敤琛板噺姝ラ暱銆佹寚鏁扮Щ鍔ㄥ钩鍧囧拰鏁板€艰鍓?
- 绉婚櫎鎵€鏈?`shifting[err_code] += ...shifting_rate...` 鐨勭‖缂栫爜閫昏緫

### Documentation
- 鏇存柊 `kl8_running.py` 鐨勪娇鐢ㄨ鏄庡拰鍙傛暟瑙ｉ噴

## [1.2.0] - 2025-10-12 鐗瑰緛澧炲己寮曟搸
### Added
- 鏂板 `src/analysis/feature_enhancer.py`锛屾彁渚涜繎鏈熷姩閲忎笌鍏辩幇璋辩殑娣峰悎璇勫垎銆?
- `kl8_analysis.py` / `kl8_analysis_plus.py` 鏀寔 `--feature_mode` 鍙傛暟锛坄hybrid` / `momentum` / `cooccurrence`锛夈€?
- 鏂板 `tests/test_feature_enhancer.py`锛岃鐩栫┖鏁版嵁銆佸姩閲忎笌鍏辩幇寰楀垎璁＄畻銆?

### Changed
- 楂樼骇鍙风爜鐢熸垚娴佺▼鍙犲姞鐗瑰緛寰楀垎锛屽湪鍊欓€夌瓫閫夐樁娈靛姞鍏ュ姞鏉冭瘎鍒嗐€?
- Plus 鐗堟湰鍦ㄨ礉鍙舵柉鍊欓€夋帓搴忎笌琛ュ叏鏃跺紩鐢ㄧ壒寰佸緱鍒嗭紝閬垮厤鍗曠函闅忔満琛ヤ綅銆?

### Documentation
- 閲嶅啓 `README.md`銆乣docs/kl8_usage_guide.md`銆乣docs/kl8_algorithm_theory.md`銆乣docs/api.md`锛屽姞鍏ョ壒寰佸寮鸿鏄庝笌绀轰緥銆?
- `docs/decision_record.md` 璁板綍鐗瑰緛澧炲己寮曟搸鍙栬垗銆?

## [1.1.0] - 2025-12-19 馃殌 澶氱嚎绋嬫灦鏋勪紭鍖?
### Added
- **澶氱嚎绋嬩紭鍖栫増鏈?*锛氭柊澧?`kl8_analysis_plus.py` 鍜?`kl8_cash_plus.py`
- **鏁版嵁涓嬭浇浼樺寲**锛氬疄鐜?`download_data_if_needed()` 鍑芥暟锛屼富绾跨▼鍗曟涓嬭浇
- **绾跨▼瀹夊叏鏈哄埗**锛氫娇鐢?`threading.Lock` 淇濇姢鍏变韩璧勬簮璁块棶
- **ThreadPoolExecutor鏋舵瀯**锛氭浛浠ｅ杩涚▼锛岄伩鍏嶅叏灞€鍙橀噺鍐茬獊
- **鏅鸿兘閿欒澶勭悊**锛氬崟绾跨▼澶辫触涓嶅奖鍝嶆暣浣撴祦绋?
- **瀹炴椂杩涘害鐩戞帶**锛氳缁嗙殑澶勭悊鐘舵€佸拰鎬ц兘鎸囨爣鏄剧ず
- **鍐呭瓨浼樺寲**锛氱嚎绋嬪叡浜唴瀛橈紝闄嶄綆60%鍐呭瓨鍗犵敤

### Changed
- **骞跺彂妯″瀷鍗囩骇**锛氫粠 `multiprocessing.Process` 杩佺Щ鍒?`concurrent.futures.ThreadPoolExecutor`
- **鍏ㄥ眬鍙橀噺澶勭悊**锛氬伐浣滅嚎绋嬩娇鐢ㄥ眬閮ㄥ彉閲忥紝閬垮厤鐘舵€佸啿绐?
- **鏂囦欢鍚嶈В鏋愬寮?*锛氭敮鎸?"next" 鍜岄潪鏁板瓧鏈熷彿鏍囪瘑绗?
- **閿欒闅旂鏀硅繘**锛氬紓甯哥嚎绋嬩笉褰卞搷鍏朵粬宸ヤ綔绾跨▼鎵ц

### Performance
- **缃戠粶璇锋眰浼樺寲**锛氬噺灏?0%閲嶅鏁版嵁涓嬭浇
- **澶勭悊閫熷害鎻愬崌**锛氬ぇ瑙勬ā鎵归噺澶勭悊鎬ц兘鎻愬崌40%
- **鍐呭瓨浣跨敤浼樺寲**锛氬嘲鍊煎唴瀛樺崰鐢ㄩ檷浣?0%
- **绋冲畾鎬ф彁鍗?*锛氬绾跨▼閿欒鐜囬檷浣?0%

### Documentation
- **鏋舵瀯鏂囨。鏇存柊**锛氭坊鍔犲绾跨▼浼樺寲鏋舵瀯鍥惧拰鎬ц兘瀵规瘮
- **浣跨敤鎸囧崡澧炲己**锛氱獊鍑篜lus鐗堟湰浼樺寲鐗规€у拰浣跨敤寤鸿
- **杩愮淮鎸囧崡鎵╁睍**锛氭柊澧炲绾跨▼鐩戞帶鍜屾晠闅滄帓鏌ョ珷鑺?
- **API鏂囨。瀹屽杽**锛氳ˉ鍏呯嚎绋嬪畨鍏ˋPI浣跨敤璇存槑鍜岀ず渚嬩唬鐮?

---

## [1.0.0] - 2025-10-11
### Added
- 绮剧畝鍚庣殑 `src/common.py`銆乣src/config.py`銆乣src/data_fetcher.py` 浠ュ強瀵瑰簲鍗曞厓娴嬭瘯銆?
- Makefile銆佺ず渚嬫暟鎹?(`data/kl8/data.csv`) 涓庡叏鏂扮殑 README/鏂囨。浣撶郴銆?
- `ASSUMPTIONS.md`銆乣docs/decision_record.md`銆乣docs/architecture.md`銆乣docs/api.md`銆乣docs/ops.md`銆乣agent_report.md`銆?

### Changed
- 鍙繚鐣欏揩涔?8 鐜╂硶锛屾墍鏈夊叕鍏辨帴鍙ｉ粯璁ら拡瀵?`kl8`銆?
- 閲嶆柊閰嶇疆娴嬭瘯娴佺▼锛屼娇 `pytest --cov=src` 搴旂敤浜庢牳蹇冩ā鍧椼€?
- `scripts/get_data.py` 鍜?`examples/analysis_example.py` 涓庢柊缁撴瀯瀵归綈銆?

### Removed
- pipeline銆乵odeling銆乸reprocessing 绛変笌妯″瀷璁粌鐩稿叧鐨勪唬鐮佸強娴嬭瘯銆?
