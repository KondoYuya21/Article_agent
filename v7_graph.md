```mermaid
graph TD;
    START --> search;
    search --> router;
    router -->批評エージェント;
    router -->ファクトエージェント;
    批評エージェント --> router;
    ファクトエージェント --> router;
    router -->END;
```


