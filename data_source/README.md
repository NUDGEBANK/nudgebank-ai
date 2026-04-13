# Prediction Source Tables

다음 달 총 소비금액 예측 데이터셋 생성을 위해 아래 CSV들을 이 폴더에 둡니다.

권장 흐름:

1. DB에서 Python export 스크립트로 CSV 생성
2. 생성된 CSV를 눈으로 검수
3. CSV 기반으로 feature dataset 생성
4. XGBoost 학습 및 추론

- `card_transaction.csv`
- `consumer_baseline.csv`
- `consumer_monthly_analysis.csv`
- `age_group_baseline.csv`

최소 필수 파일은 `card_transaction.csv`입니다.

DB에서 바로 CSV를 생성하려면:

```powershell
python model\export_source_tables.py
```

현재 bank-backend 기준 DB는 PostgreSQL입니다.

로컬 개발에서는 아래 위치의 `.env`도 자동으로 읽습니다.

- `nudgebank-ai/.env`
- `../bank-backend/.env`

단, 배포 환경에서는 `.env`보다 실제 환경변수가 우선이며, 비밀값은 커밋하지 않는 것을 권장합니다.

기본 export 대상:

- `card_transaction.csv`
- `consumer_baseline.csv`
- `age_group_baseline.csv`
- `member.csv`
- `account.csv`
- `market_category.csv`

`card_transaction.csv`는 단순 `SELECT *`가 아니라 아래 정보를 join해서 export합니다.

- `consumer_id`
- `approved_at`
- `amount`
- `market_name`
- `category_name`
- `mcc`
- `sex`
- `birth`

예상 join 기준:

- `consumer_id` 또는 `member_id` 또는 `user_id`
- `year_month` 또는 `target_month` 또는 `analysis_month`

`card_transaction.csv` 예상 주요 컬럼 예시:

- `consumer_id`
- `approved_at`
- `amount`
- `card_tpbuz_nm_1`
- `card_tpbuz_nm_2`
- `spending_type` (있으면 재사용, 없으면 백엔드에서 분류 후 export 권장)

기본 실행:

```powershell
python model\export_source_tables.py
python model\consumption_dataset.py
python model\train_consumption_xgboost.py
python model\predict_next_month_consumption.py
```
