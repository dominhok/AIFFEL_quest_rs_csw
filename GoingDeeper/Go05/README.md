# AIFFEL Campus Online Code Peer Review Templete
- 코더 : 조성우
- 리뷰어 : 김민호


# PRT(Peer Review Template)
- [X]  **1. 주어진 문제를 해결하는 완성된 코드가 제출되었나요?**
    - 문제에서 요구하는 최종 결과물이 첨부되었는지 확인
        - ![image](https://github.com/user-attachments/assets/9739ae0d-0cfe-4eac-9239-b05785e413f3)
        - KITTI 데이터셋에 대해 분석하고 바운딩 박스 시각화까지 진행하셨다.
        - ![image](https://github.com/user-attachments/assets/0b25b488-f697-46c9-85d6-306a1e393bf7)
        - 정상적으로 학습이 진행되었고, Validation Loss를 기준으로 가장 좋은 체크포인트를 찾아서 실험에 사용하셨다.
        - ![image](https://github.com/user-attachments/assets/b452906f-4532-441b-a547-bc575597a340)
        - 루브릭에서 요구한 90점을 넘는 결과를 달성하셨다.
        - 루브릭에서 요구한 3가지를 모두 잘 구현해 주셨다.

    
- [X]  **2. 전체 코드에서 가장 핵심적이거나 가장 복잡하고 이해하기 어려운 부분에 작성된 
주석 또는 doc string을 보고 해당 코드가 잘 이해되었나요?**
    - 해당 코드 블럭을 왜 핵심적이라고 생각하는지 확인
    - 해당 코드 블럭에 doc string/annotation이 달려 있는지 확인
    - 해당 코드의 기능, 존재 이유, 작동 원리 등을 기술했는지 확인
    - 주석을 보고 코드 이해가 잘 되었는지 확인
        - ![image](https://github.com/user-attachments/assets/40a9abd7-e9fd-4abd-af4f-2496ac591b33)
        - ![image](https://github.com/user-attachments/assets/b6912116-d7c8-47e9-b905-a90939758df2)
        - 정지조건을 구현해 주신 부분에서 보조 로직과 관련한 주석을 남겨주셔서 코드를 이해하기 쉬웠다.

        
- [X]  **3. 에러가 난 부분을 디버깅하여 문제를 해결한 기록을 남겼거나
새로운 시도 또는 추가 실험을 수행해봤나요?**
    - 문제 원인 및 해결 과정을 잘 기록하였는지 확인
    - 프로젝트 평가 기준에 더해 추가적으로 수행한 나만의 시도, 
    실험이 기록되어 있는지 확인
        - ![image](https://github.com/user-attachments/assets/7e3fe2aa-6e83-423a-889f-12e4ea813652)
        - Precision과 Recall 같은 Metric을 이용하시기 위한 시도중, 실제 객체 수와 예측한 객체 수가 달라 더이상 진행할 수 없는 지점에서 에러 확인을 위해 로그를 찍는 방식으로 문제를 해결하고자 한 기록을 남겨주셨다.
        
- [X]  **4. 회고를 잘 작성했나요?**
    - 주어진 문제를 해결하는 완성된 코드 내지 프로젝트 결과물에 대해
    배운점과 아쉬운점, 느낀점 등이 기록되어 있는지 확인
    - 전체 코드 실행 플로우를 그래프로 그려서 이해를 돕고 있는지 확인
        - ![image](https://github.com/user-attachments/assets/2bca4a88-e5a8-46ee-84cf-94aa25848ef5)
        - 회고 역시 잘 작성해주셨다. 전체적으로 중복 레이블 문제를 수정하고자 했던 노력과, mAP로 평가하고자 하신 노력이 보인 코드였다.

        
- [X]  **5. 코드가 간결하고 효율적인가요?**
    - 파이썬 스타일 가이드 (PEP8) 를 준수하였는지 확인
    - 코드 중복을 최소화하고 범용적으로 사용할 수 있도록 함수화/모듈화했는지 확인
        - ![image](https://github.com/user-attachments/assets/08e385ab-491d-4066-9acb-bee1031cae1d)
        - 전체적으로 함수화가 잘 되어있고, 간결한 코드로 효율적이게 프로젝트를 진행해주셨다.



# 회고(참고 링크 및 코드 개선)
```
시간이 조금 더 있었으면 마지막 mAP 실험까지 충분히 진행하셨을 것 같다. 나는 Metric을 도입하지 않고 그저 자동 라벨링 후
샘플 개수만 늘리는 실험만 진행했는데 여러 metric을 도입하시는 과정을 볼 수 있어서 배울 점이 많은 코드였다.
```
