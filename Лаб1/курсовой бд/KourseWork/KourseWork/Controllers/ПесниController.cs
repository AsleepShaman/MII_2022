using System;
using System.Collections.Generic;
using System.Data;
using System.Data.Entity;
using System.Linq;
using System.Net;
using System.Web;
using System.Web.Mvc;
using KourseWork.Models;

namespace KoursWork.Controllers
{
    public class ПесниController : Controller
    {
        private AudioStudioDBEntities2 db = new AudioStudioDBEntities2();

        // GET: Песни
        public ActionResult Index()
        {
            var песни = db.Песни.Include(п => п.Альбомы);
            return View(песни.ToList());
        }
        public ActionResult UserIndex(string SongName)
        {
            var песни = db.Песни.Include(п => п.Альбомы).AsEnumerable();
            if (!string.IsNullOrEmpty(SongName))
            {
                песни = db.Песни.Where(s => s.Название_песни == SongName);
            }
            return View(песни.ToList());
        }
        public ActionResult GuestIndex()
        {
            var песни = db.Песни.Include(п => п.Альбомы);
            return View(песни.ToList());
        }
        [HttpPost]
        public ActionResult Index(string SongName)
        {
            IEnumerable<Песни> AllSongs = db.Песни.AsEnumerable();
            if (!string.IsNullOrEmpty(SongName))
            {
                AllSongs = db.Песни.Where(s => s.Название_песни == SongName);
            }
            return View(AllSongs.ToList());
        }
        public ActionResult Error()
        {
            return View();
        }

        // GET: Песни/Details/5
        public ActionResult Details(int? id)
        {
            if (id == null)
            {
                return RedirectToAction("Error404", "Error");
            }
            Песни песни = db.Песни.Find(id);
            if (песни == null)
            {
                return RedirectToAction("Error404", "Error");
            }
            return View(песни);
        }

        // GET: Песни/Create
        public ActionResult Create()
        {
            ViewBag.Номер_альбома = new SelectList(db.Альбомы, "Номер_альбома", "Название_альбома");
            return View();
        }

        // POST: Песни/Create
        // Чтобы защититься от атак чрезмерной передачи данных, включите определенные свойства, для которых следует установить привязку. Дополнительные 
        // сведения см. в разделе https://go.microsoft.com/fwlink/?LinkId=317598.
        [HttpPost]
        [ValidateAntiForgeryToken]
        public ActionResult Create([Bind(Include = "Номер_песни,Название_песни,Битрейт,Длина,Номер_альбома,Сопр__инструменты")] Песни песни)
        {
            if (ModelState.IsValid)
            {
                try
                {
                    db.Песни.Add(песни);
                    db.SaveChanges();
                    return RedirectToAction("Index");
                }
                catch
                {
                    ViewBag.Data = "Error!";
                    return RedirectToAction("Error");
                }
            }

            ViewBag.Номер_альбома = new SelectList(db.Альбомы, "Номер_альбома", "Название_альбома", песни.Номер_альбома);
            return View(песни);
        }

        // GET: Песни/Edit/5
        public ActionResult Edit(int? id)
        {
            if (id == null)
            {
                return RedirectToAction("Error404", "Error");
            }
            Песни песни = db.Песни.Find(id);
            if (песни == null)
            {
                return RedirectToAction("Error404", "Error");
            }
            ViewBag.Номер_альбома = new SelectList(db.Альбомы, "Номер_альбома", "Название_альбома", песни.Номер_альбома);
            return View(песни);
        }

        // POST: Песни/Edit/5
        // Чтобы защититься от атак чрезмерной передачи данных, включите определенные свойства, для которых следует установить привязку. Дополнительные 
        // сведения см. в разделе https://go.microsoft.com/fwlink/?LinkId=317598.
        [HttpPost]
        [ValidateAntiForgeryToken]
        public ActionResult Edit([Bind(Include = "Номер_песни,Название_песни,Битрейт,Длина,Номер_альбома,Сопр__инструменты")] Песни песни)
        {
            if (ModelState.IsValid)
            {
                try{db.Entry(песни).State = EntityState.Modified;
                db.SaveChanges();
                return RedirectToAction("Index");
                }
                catch
                {
                    ViewBag.Data = "Ошибка редактирования!";
                    return RedirectToAction("Error");
                }
            }
            ViewBag.Номер_альбома = new SelectList(db.Альбомы, "Номер_альбома", "Название_альбома", песни.Номер_альбома);
            return View(песни);
        }

        // GET: Песни/Delete/5
        public ActionResult Delete(int? id)
        {
            if (id == null)
            {
                return RedirectToAction("Error404", "Error");
            }
            Песни песни = db.Песни.Find(id);
            if (песни == null)
            {
                return RedirectToAction("Error404", "Error");
            }
            return View(песни);
        }

        // POST: Песни/Delete/5
        [HttpPost, ActionName("Delete")]
        [ValidateAntiForgeryToken]
        public ActionResult DeleteConfirmed(int id)
        {
            Песни песни = db.Песни.Find(id);
            db.Песни.Remove(песни);
            db.SaveChanges();
            return RedirectToAction("Index");
        }

        protected override void Dispose(bool disposing)
        {
            if (disposing)
            {
                db.Dispose();
            }
            base.Dispose(disposing);
        }
    }
}
