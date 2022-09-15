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
    public class ИсполнительController : Controller
    {
        private AudioStudioDBEntities2 db = new AudioStudioDBEntities2();

        // GET: Исполнитель
        public ActionResult Index()
        {
            return View(db.Исполнитель.ToList());
        }
        public ActionResult Error()
        {
            return View();
        }

        public ActionResult UserIndex()
        {
            return View(db.Исполнитель.ToList());
        }
        public ActionResult GuestIndex()
        {
            return View(db.Исполнитель.ToList());
        }

        // GET: Исполнитель/Details/5
        public ActionResult Details(string id)
        {
            if (id == null)
            {
                return RedirectToAction("Error404", "Error");
            }
            Исполнитель исполнитель = db.Исполнитель.Find(id);
            if (исполнитель == null)
            {
                return RedirectToAction("Error404", "Error");
            }
            return View(исполнитель);
        }

        // GET: Исполнитель/Create
        public ActionResult Create()
        {
            return View();
        }

        // POST: Исполнитель/Create
        // Чтобы защититься от атак чрезмерной передачи данных, включите определенные свойства, для которых следует установить привязку. Дополнительные 
        // сведения см. в разделе https://go.microsoft.com/fwlink/?LinkId=317598.
        [HttpPost]
        [ValidateAntiForgeryToken]
        public ActionResult Create([Bind(Include = "Паспорт_Исполнителя,ФИО,Кол_во_альбомов,Карьера,Кол_во_наград,Страна,Эл__почта")] Исполнитель исполнитель)
        {
            if (ModelState.IsValid)
            {
                try
                {
                    db.Исполнитель.Add(исполнитель);
                    db.SaveChanges();
                    return RedirectToAction("Index");
                }
                catch
                {
                    ViewBag.Data = "Error!";
                    return RedirectToAction("Error");
                }
            }

            return View(исполнитель);
        }

        // GET: Исполнитель/Edit/5
        public ActionResult Edit(string id)
        {
            if (id == null)
            {
                return RedirectToAction("Error404", "Error");
            }
            Исполнитель исполнитель = db.Исполнитель.Find(id);
            if (исполнитель == null)
            {
                return RedirectToAction("Error404", "Error");
            }
            return View(исполнитель);
        }

        // POST: Исполнитель/Edit/5
        // Чтобы защититься от атак чрезмерной передачи данных, включите определенные свойства, для которых следует установить привязку. Дополнительные 
        // сведения см. в разделе https://go.microsoft.com/fwlink/?LinkId=317598.
        [HttpPost]
        [ValidateAntiForgeryToken]
        public ActionResult Edit([Bind(Include = "Паспорт_Исполнителя,ФИО,Кол_во_альбомов,Карьера,Кол_во_наград,Страна,Эл__почта")] Исполнитель исполнитель)
        {
            if (ModelState.IsValid)
            {
                try{db.Entry(исполнитель).State = EntityState.Modified;
                db.SaveChanges();
                return RedirectToAction("Index");
                }
                catch
                {
                    ViewBag.Data = "Ошибка редактирования!";
                    return RedirectToAction("Error");
                }
            }
            return View(исполнитель);
        }

        // GET: Исполнитель/Delete/5
        public ActionResult Delete(string id)
        {
            if (id == null)
            {
                return RedirectToAction("Error404", "Error");
            }
            Исполнитель исполнитель = db.Исполнитель.Find(id);
            if (исполнитель == null)
            {
                return RedirectToAction("Error404", "Error");
            }
            return View(исполнитель);
        }

        // POST: Исполнитель/Delete/5
        [HttpPost, ActionName("Delete")]
        [ValidateAntiForgeryToken]
        public ActionResult DeleteConfirmed(string id)
        {
            try{
                Исполнитель исполнитель = db.Исполнитель.Find(id);
            db.Исполнитель.Remove(исполнитель);
            db.SaveChanges();
            return RedirectToAction("Index");
            }
            catch
            {
                ViewBag.Data = "Ошибка удаления! Возможно, эти данные где-то есть...";
                return RedirectToAction("Error");
            }
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
