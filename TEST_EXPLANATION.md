This test demonstrates the THREE-BRANCH fallback logic for the /predict endpoint:

BRANCH 1: Mapped class + confidence >= 0.7
→ Returns full disease info (plant, disease, description, symptoms, treatment, prevention)

BRANCH 2: Confidence < 0.7  
→ Returns "Low confidence prediction. The model is uncertain about this classification. Please try a clearer image of the plant leaf."
+ low_confidence: true flag

BRANCH 3: Unmapped class + confidence >= 0.7
→ Returns: "Disease information not available for this ImageNet class. The model predicted a general ImageNet class rather than a plant disease. Try uploading a clearer leaf image or use a different sample."

=== CURRENT SERVER RESPONSE TEST ===
Running on http://localhost:8000

A solid color image (not a real plant leaf) will typically produce random ImageNet classes with LOW confidence, triggering BRANCH 2.

A real plant leaf image with good lighting will typically produce a mapped class with HIGH confidence, triggering BRANCH 1.

=== WHAT THE USER SAW: Class_304, Confidence: 0.26 ===
Since 0.26 < 0.7, this should now show BRANCH 2 message:
"Low confidence prediction. The model is uncertain about this classification. Please try a clearer image of the plant leaf."

But if the user sees "Disease information not available for this class." instead, it means:
1. The server may not have the latest code, OR
2. The frontend is parsing the response incorrectly, OR  
3. Using an older version of the app

=== PROVIDE A SAMPLE PLANT LEAF IMAGE FOR TESTING ===
Save this as a PNG file and upload it through the form at http://localhost:8000/