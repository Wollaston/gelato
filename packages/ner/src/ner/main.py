import requests
from dataclasses import dataclass
from enum import StrEnum
import logging

logger = logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


class BillType(StrEnum):
    HR = "Hr"
    S = "S"
    HJRES = "Hjres"
    SJRES = "Sjres"
    HCONRES = "Hconres"
    SCONRES = "Sconres"
    HRES = "Hres"
    SRES = "Sres"


class Model(StrEnum):
    XlmrBase = "xlm-r-base"
    RobertaBase = "roberta-base"
    BertBaseCased = "bert-base-cased"


@dataclass()
class CDG:
    """Helper class to interface with CDG NER Backend"""

    backend = "http://cdg-ner-backend.flycast:80"
    db = "http://cdg-ner-db.flycast:80"

    def healthcheck_backend(self):
        """Confirms the backend is reachable"""
        try:
            res = requests.get(self.backend + "/health")
            match res.status_code:
                case 200:
                    logging.info("Succesfully connected to Backend")
                case _:
                    logging.error("Error connecting to Backend")
        except requests.exceptions.RequestException as e:
            logging.error(
                "Make sure you are connected to the cdg-ner Fly.io network via WireGuard"
            )
            raise SystemExit(e)

    def healthcheck_models(self):
        """Confirms the models are reachable"""
        try:
            res = requests.get(self.backend + "/model/health/" + Model.XlmrBase)
            match res.status_code:
                case 200:
                    logging.info("Succesfully connected to XLM-R Base model")
                case _:
                    logging.error("Error connecting to XLM-R Base Model")
            res = requests.get(self.backend + "/model/health/" + Model.RobertaBase)
            match res.status_code:
                case 200:
                    logging.info("Succesfully connected to XLM-R Large Conll model")
                case _:
                    logging.error("Error connecting to XLM-R Large Conll Model")
            res = requests.get(self.backend + "/model/health/" + Model.BertBaseCased)
            match res.status_code:
                case 200:
                    logging.info("Succesfully connected to Bert Base Cased model")
                case _:
                    logging.error("Error connecting to Bert Base Cased model")
        except requests.exceptions.RequestException as e:
            logging.error(
                "Make sure you are connected to the cdg-ner Fly.io network via WireGuard"
            )
            raise SystemExit(e)

    def healthcheck_db(self):
        """Confirms the db is reachable"""
        try:
            res = requests.get(self.db + "/health")
            match res.status_code:
                case 200:
                    logging.info("Succesfully connected to DB")
                case _:
                    logging.error("Error connecting to DB")
        except requests.exceptions.RequestException as e:
            logging.error(
                "Make sure you are connected to the cdg-ner Fly.io network via WireGuard"
            )
            raise SystemExit(e)

    def get_bill(
        self, congress: int, bill_type: BillType, number: int
    ) -> dict[str, object] | None:
        """Gets the requested Bill from the backend"""
        try:
            res = requests.get(self.backend + f"/bill/{congress}/{bill_type}/{number}")
            match res.status_code:
                case 200:
                    return res.json()
                case _:
                    logging.error(res.text)
        except requests.exceptions.RequestException as e:
            logging.error(
                "Make sure you are connected to the cdg-ner Fly.io network via WireGuard"
            )
            raise SystemExit(e)

    def prediction(self, text: str, model: Model) -> dict[str, object] | None:
        """Predicts the BIO-encoding of the provided text parameter"""
        try:
            res = requests.post(
                self.backend + f"/model/predict/{model}", json={"text": text}
            )
            match res.status_code:
                case 200:
                    return res.json()
                case _:
                    logging.error(res.text)
        except requests.exceptions.RequestException as e:
            logging.error(
                "Make sure you are connected to the cdg-ner Fly.io network via WireGuard"
            )
            raise SystemExit(e)


if __name__ == "__main__":
    cdg = CDG()
    cdg.healthcheck_backend()
    cdg.healthcheck_db()
    cdg.healthcheck_models()
    bill = cdg.get_bill(119, BillType.HR, 1897)
    if bill:
        print(bill["title"])
    prediction = cdg.prediction(
        "The Golden State Warriors are an American professional basketball team based in San Francisco.",
        Model.RobertaBase,
    )
    if prediction:
        print(prediction)
