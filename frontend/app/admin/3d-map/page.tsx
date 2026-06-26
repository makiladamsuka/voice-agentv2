import dynamic from "next/dynamic";
import fs from "fs";
import path from "path";

const NavigationMap = dynamic(() => import("@/components/app/isometric-map"), {
  ssr: false,
});

// Demo page: load Floor 1 data and display the map
async function loadFloorData() {
  try {
    const filePath = path.join(
      process.cwd(),
      "../backend/data/map_graph_floor_1.json",
    );
    const data = JSON.parse(fs.readFileSync(filePath, "utf-8"));
    return data;
  } catch {
    return null;
  }
}

export default async function ThreeDMapViewer() {
  const data = await loadFloorData();

  const demoPath =
    data?.nodes
      ?.slice(0, 2)
      .map((n: { x: number; z: number; building: string; size?: number[] }) => {
        const b = data.buildings?.[n.building] || { position: [0, 0, 0] };
        return [b.position[0] + n.x, 0.3, b.position[2] + n.z];
      }) || [];

  return (
    <div className="w-full h-screen">
      <NavigationMap
        path={demoPath}
        nodes={(data?.nodes || []).map(
          (n: { x: number; z: number; building: string; size?: number[] }) => {
            const b = data?.buildings?.[n.building] || { position: [0, 0, 0] };
            return {
              ...n,
              world: [
                b.position[0] + n.x,
                (n.size?.[1] || 1) / 2,
                b.position[2] + n.z,
              ],
            };
          },
        )}
        buildings={data?.buildings || {}}
        destination={data?.nodes?.[0]?.label || "Unknown"}
      />
    </div>
  );
}
