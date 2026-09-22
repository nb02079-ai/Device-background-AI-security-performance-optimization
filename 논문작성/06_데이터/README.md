# 06_데이터 — 데이터 보관 안내

원본 데이터는 용량이 커서 정리 zip에 포함하지 않았다. 아래 표를 기준으로 로컬에서 배치한다.

| 파일 | 용량 | 권장 위치 | 비고 |
|---|---|---|---|
| `dmbd_event_counts.csv` | 수 MB | **이 폴더** | DMBD 실행 단위 집계본. **모든 실험의 입력**. 가장 중요 |
| `dmbd_counts_nolabel.csv` | 수 MB | 이 폴더 (선택) | 라벨 결합 전 집계본. 위 파일이 있으면 불필요 |
| `MalMem2022.csv` | 약 19.6 MB | 이 폴더 | CIC-MalMem-2022 (Filename 컬럼 포함판). 4.1 사례연구용 |
| `Obfuscated-MalMem2022.csv` | 약 19 MB | 선택 | Filename 없는 판. `MalMem2022.csv`가 있으면 불필요 |
| `trees_0.json` ~ `trees_7.json` | **약 8 GB** | **외장 드라이브 등 별도 위치** | DMBD 원시 이벤트 로그. 재집계할 때만 필요 |
| `truth_labels_train.json` / `truth_labels_test.json` | 작음 | 원시 로그와 함께 | DMBD 라벨 |

**원시 로그(8GB)를 논문 폴더에 두지 않는 이유**: 백업·동기화가 무거워지고, 실험은 집계본(`dmbd_event_counts.csv`)만으로 전부 재현된다.

**출처**
- DMBD 2025: Lawrence Livermore National Laboratory, gdo-wintap.llnl.gov → "Access Data Here"
- CIC-MalMem-2022: UNB CIC 공식 페이지, 또는 Kaggle 미러
