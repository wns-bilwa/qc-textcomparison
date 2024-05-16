import logging
import azure.functions as func
import json
from shared_code.create_embedding import (
    generate_embeddings,
    cosine_similarity,
    fuzzyratio_similarity,
)

app = func.FunctionApp(http_auth_level=func.AuthLevel.FUNCTION)


@app.route(route="http_trigger_textcomparison", methods=("POST",))
def http_trigger_textcomparison(req: func.HttpRequest) -> func.HttpResponse:
    logging.info("Python HTTP trigger function processed a request.")
    try:
        req_body = req.get_json()
    except Exception as e:
        logging.error("Error in fetching value : %s", e)
        return func.HttpResponse(body=f"Error in fetching value {e}", status_code=500)
    QCData = req_body.get("InvoiceDatalist")
    response= json.dumps(qc_textcomparison(QCData))
    
    return func.HttpResponse(response, headers={"Content-type": "application/json"})


def qc_textcomparison(QCLIST):
    finalJsonResponse = {
        "invoiceList": [],
    }
    for invoice in QCLIST:

        S1 = invoice['auditSheet']
        S2 = invoice['invoiceData']
        METHOD = invoice['method']
        FIELDNAME = invoice['fieldName']

        fileResponse = {
                "fieldName": FIELDNAME,
                "auditSheet": S1,
                "invoiceData": S2,
                "fuzzyconfidenceScore":"",
                "embeddingconfidenceScore":"",
                "error": ""
            }

        if METHOD == "both":
            fuzzy_similarityscore = fuzzyratio_similarity(S1, S2)
            print("Fuzzy Ratio:", fuzzy_similarityscore)

            embed1 = generate_embeddings(S1)
            embed2 = generate_embeddings(S2)
            embedding_similarityscore = cosine_similarity(embed1, embed2)
            print("Cosine Similarity Score: ", embedding_similarityscore * 100)

            fileResponse["fuzzyconfidenceScore"] = fuzzy_similarityscore
            fileResponse["embeddingconfidenceScore"] = embedding_similarityscore

        else:
            embed1 = generate_embeddings(S1)
            embed2 = generate_embeddings(S2)
            similarityscore = cosine_similarity(embed1, embed2)
            fuzz_similarityscore = fuzzyratio_similarity(S1, S2)
            similarityscore = ((similarityscore * 100) + fuzz_similarityscore) / 2
            print("Hybrid Ratio:", similarityscore)

            fileResponse["fuzzyconfidenceScore"] = similarityscore
        
        finalJsonResponse["invoiceList"].append(fileResponse)

    return finalJsonResponse


    
