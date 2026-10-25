import json
import numpy as np
from PIL import Image
from yabitsu import kn5, visual_updates
from yabitsu.manifest import Manifest


def test_visual_update_preserves_uv_and_keeps_old_collision(tmp_path):
    source=tmp_path/'source'; m=Manifest(source)
    (source/'textures').mkdir()
    Image.new('RGB',(8,8),'gray').save(source/'textures/wall.png')
    m.add_material('concrete',texture='textures/wall.png')
    v=np.array([[0,0,0],[2,0,0],[2,0,3],[0,0,3]],float)
    uv=np.array([[0,0],[0,.5],[1.5,.5],[1.5,0]],float)
    f=np.array([[0,1,2,3]])
    m.add_mesh('base','1WALL_kitahara_obj_retaining',v,f,uv,'concrete')
    m.add_mesh('base','1ROAD_unchanged',v,f,uv,'concrete')
    m.save(); original=(source/'manifest.json').read_bytes()
    p=visual_updates.build(source,tmp_path/'out',wall_texture=source/'textures/wall.png')
    assert (source/'manifest.json').read_bytes()==original
    k=kn5.read_kn5(p)
    nodes=k['root']['children']; assert len(nodes)==1
    assert nodes[0]['name']=='visual_1WALL_kitahara_obj_retaining'
    np.testing.assert_allclose(nodes[0]['vertices'][:,6:8],np.c_[uv[:,0],1-uv[:,1]])
    ext=(p.parent/'extension/visual_updates.ini').read_text()
    assert 'IS_RENDERABLE=0' in ext and 'IS_ACTIVE=0' not in ext
    assert '1ROAD_unchanged' not in ext
    out=json.loads((p.parent/'manifest.json').read_text())
    assert out['meshes'][nodes[0]['name']]['material']=='visual_kitahara_wall'
