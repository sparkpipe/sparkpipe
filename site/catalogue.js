window.SP_CATALOGUE = {
  "items": [
    {
      "id": "sparkpipe",
      "name": "SparkPipe",
      "nameKo": "SparkPipe",
      "publisher": "HYOR AI Tech Ltd.",
      "kind": "engine",
      "summary": "Our inference engine, with its API and Functions. It runs the open-source models listed on the Fleet List on all the machines of your fleet, and your tools reach it through the Anthropic Messages API, the OpenAI Responses API and OpenAI chat completions.",
      "summaryKo": "저희 추론 엔진과 그 API 및 기능. 플릿 목록에 게시된 오픈소스 모델을 플릿의 모든 장비에서 실행하며, 고객의 도구는 Anthropic Messages API, OpenAI Responses API, OpenAI chat completions로 엔진을 사용합니다.",
      "license": {
        "name": "Functional Source License 1.1 (ALv2 future license) with an additional permission",
        "nameKo": "Functional Source License 1.1(ALv2 장래 라이선스) 및 추가 허가",
        "texts": [
          "licenses/sparkpipe/LICENSE.txt",
          "licenses/sparkpipe/LICENSE-PERMISSION.txt"
        ]
      },
      "acceptance": true
    },
    {
      "id": "vllm",
      "name": "vLLM",
      "nameKo": "vLLM",
      "publisher": "vLLM project",
      "kind": "engine",
      "summary": "An open-source inference engine published by the vLLM project. It installs the vLLM release from its publisher into your account on all the machines of your fleet. You choose the models it runs and obtain each from its publisher, under its license.",
      "summaryKo": "vLLM 프로젝트가 배포하는 오픈소스 추론 엔진. 그 배포자의 vLLM 릴리스를 플릿의 모든 장비에 있는 고객 계정에 설치합니다. 실행할 모델은 고객이 고르며, 각 모델은 그 배포자로부터 그 라이선스에 따라 받습니다.",
      "license": {
        "name": "Apache License 2.0",
        "nameKo": "Apache License 2.0",
        "texts": [
          "licenses/vllm/LICENSE.txt"
        ]
      },
      "acceptance": false
    },
    {
      "id": "none",
      "name": "Nothing: keep the base image",
      "nameKo": "설치하지 않음: 기본 이미지 유지",
      "publisher": "HYOR AI Tech Ltd.",
      "kind": "none",
      "summary": "Installs nothing. The fleet keeps only our base image and your own working state, and you install what you run yourself.",
      "summaryKo": "아무것도 설치하지 않습니다. 플릿에는 저희 기본 이미지와 고객 자신의 작업 상태만 남으며, 실행할 것은 고객이 직접 설치합니다.",
      "license": null,
      "acceptance": false
    }
  ]
};
