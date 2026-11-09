"""Update SEQ fields via Word COM, save docx and export PDF."""
import os
import win32com.client as win32

DOC = r"E:\TGV inspection detection\TGV通孔检测技术预研报告.docx"
PDF = r"E:\TGV inspection detection\Exp_Data\TGV_yolo_Dataset\outputs\report_check.pdf"

word = win32.gencache.EnsureDispatch("Word.Application")
word.Visible = False
word.DisplayAlerts = 0
doc = word.Documents.Open(DOC, ReadOnly=False)
try:
    for story in doc.StoryRanges:
        rng = story
        while rng is not None:
            rng.Fields.Update()
            rng = rng.NextStoryRange
    doc.Fields.Update()
    doc.Save()
    doc.ExportAsFixedFormat(PDF, 17)
finally:
    doc.Close(SaveChanges=0)
    word.Quit()
print("done", os.path.getsize(PDF))
