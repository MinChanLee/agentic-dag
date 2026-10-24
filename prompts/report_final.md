아래 초안을 더 간결하게 다듬어 최종 리포트로 만드세요. 사실을 새로 추가하지 마세요.
출력 언어: {{ settings.output_language }}

초안: {{ report_draft | tojson }}

반드시 아래 JSON 형식으로 응답하세요:

```json
{"report": "최종 리포트"}
```
