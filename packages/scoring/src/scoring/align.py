def align(predicted: str, expected: str) -> str:
    predicted_iter = iter(predicted.splitlines())
    expected_iter = iter(expected.splitlines())

    updated_preds: str = ""

    pred_text_state = ""
    pred_label_state = ""

    for pred, expect in zip(predicted_iter, expected_iter):
        if pred == "":
            pred_text, pred_label = "", ""
        else:
            pred_text, pred_label = pred.split(" ")

        if expect == "":
            expect_text, _ = "", ""
        else:
            expect_text, _ = expect.split(" ")

        if pred_text != expect_text:
            print(pred_text, expect_text)
            while pred_text_state != expect_text:
                pred_text_state += pred_text
                if pred_label_state == "":
                    pred_label_state = pred_label

                if pred_text_state != expect_text:
                    pred = next(predicted_iter, None)

                    if pred == "":
                        pred_text, pred_label = "", ""
                    else:
                        pred_text, pred_label = (
                            pred.split(" ") if pred is not None else ("", "")
                        )
        else:
            pred_text_state = pred_text
            pred_label_state = pred_label

        print(pred_text_state)
        updated_preds += f"{pred_text_state} {pred_label_state}\n"
        print(updated_preds)
        pred_text_state = ""
        pred_label_state = ""

    updated_preds = updated_preds.replace("\n \n", "\n\n")
    return updated_preds
