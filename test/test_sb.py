import sys
sys.path.append("..")

from common import *

def test_sb():
    cur_path = Path().resolve()
    sb_file = cur_path/'scorboard-test.pkl'
    if sb_file.is_file():
        sb_file.unlink()

    sb = Scoreboard(sb_file, sb_len=2, sort='dec')
    sb.update({'score': 1.0, 'file': cur_path/'1.txt'})

    sb = Scoreboard(sb_file, sb_len=2, sort='dec')
    sb.update({'score': 2.0, 'file': cur_path/'2.txt'})
    assert len(sb) == 2

    sb.update({'score': 3.0, 'file': cur_path/'3.txt'})
    assert len(sb) == 2
    assert sb[0]['score'] == 3.0
    assert sb.is_full() == True

