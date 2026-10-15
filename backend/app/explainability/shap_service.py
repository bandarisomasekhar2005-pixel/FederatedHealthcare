
import numpy as np
import shap

class SHAPService:
    def __init__(self, model_service):
        self.ms=model_service
        self.background=self._background()
    def _background(self):
        import pandas as pd
        from pathlib import Path
        p=Path(__file__).resolve().parents[3]/"data"/"splits"/"global_train.csv"
        df=pd.read_csv(p).drop(columns=["target"])
        return df.values.astype(np.float32)[:20]
    def explain(self,x, top_k=8):
        x=np.asarray(x,dtype=np.float32)
        def f(z): return self.ms.predict_array(z)
        explainer=shap.Explainer(f,self.background,algorithm="permutation")
        explanation=explainer(x,max_evals=max(45,2*x.shape[1]+1))
        vals=np.asarray(explanation.values)[0]
        features=self.ms.__class__.__module__ and self.ms
        names=self.ms.FEATURE_NAMES_PROCESSED if hasattr(self.ms,"FEATURE_NAMES_PROCESSED") else None
        # use canonical processed names exposed by module
        from backend.app.ml.model_service import FEATURE_NAMES_PROCESSED
        rows=[{"feature":str(FEATURE_NAMES_PROCESSED[i]),"contribution":float(vals[i])} for i in range(len(vals))]
        rows.sort(key=lambda r:abs(r["contribution"]),reverse=True)
        return {"base_value":float(np.asarray(explanation.base_values).ravel()[0]),
                "top_features":rows[:top_k],
                "all_features":rows}
